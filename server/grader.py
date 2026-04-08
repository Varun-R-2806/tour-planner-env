"""
Deterministic grader for the Tour Planner environment.

Produces a score in [0.0, 1.0] and a structured feedback report.
All checks are programmatic — no LLM judgement involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models import PlaceCategory, TaskConfig


# ---------------------------------------------------------------------------
# Score weights (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHT_BUDGET_COMPLIANCE   = 0.20
WEIGHT_SAFETY              = 0.20
WEIGHT_FATIGUE             = 0.20
WEIGHT_CATEGORY_COVERAGE   = 0.25
WEIGHT_FINALIZATION        = 0.15


@dataclass
class GradeReport:
    """Human-readable + machine-readable result from the grader."""
    final_score:        float
    budget_score:       float
    safety_score:       float
    fatigue_score:      float
    category_score:     float
    finalization_score: float
    passed:             bool
    penalties:          List[str] = field(default_factory=list)
    bonuses:            List[str] = field(default_factory=list)
    summary:            str = ""

    def __str__(self) -> str:
        sep = "-" * 52
        lines = [
            sep,
            f"  TOUR PLANNER GRADER  |  Score Report",
            sep,
            f"  Budget compliance   {self.budget_score:.2f}  (×{WEIGHT_BUDGET_COMPLIANCE})",
            f"  Safety              {self.safety_score:.2f}  (×{WEIGHT_SAFETY})",
            f"  Fatigue management  {self.fatigue_score:.2f}  (×{WEIGHT_FATIGUE})",
            f"  Category coverage   {self.category_score:.2f}  (×{WEIGHT_CATEGORY_COVERAGE})",
            f"  Plan finalized      {self.finalization_score:.2f}  (×{WEIGHT_FINALIZATION})",
            sep,
            f"  FINAL SCORE         {self.final_score:.4f} / 1.0",
            f"  RESULT              {'PASS' if self.passed else 'FAIL'}",
            sep,
        ]
        if self.bonuses:
            lines.append("  Bonuses:")
            lines += [f"    + {b}" for b in self.bonuses]
        if self.penalties:
            lines.append("  Penalties:")
            lines += [f"    - {p}" for p in self.penalties]
        if self.summary:
            lines += [sep, f"  {self.summary}"]
        lines.append(sep)
        return "\n".join(lines)


class TourGrader:
    """
    Grades a completed episode given a state dictionary (from env.state).
    """

    _PASS_THRESHOLDS = {"easy": 0.65, "medium": 0.70, "hard": 0.80}

    def grade(self, state_dict: Dict[str, Any]) -> GradeReport:
        """Grade a state dict (as returned by TourState.model_dump())."""
        penalties: List[str] = []
        bonuses:   List[str] = []

        task_cfg_raw = state_dict.get("task_config")
        if not task_cfg_raw:
            return GradeReport(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False,
                               ["Missing task_config"], [], "Error")

        task_cfg = TaskConfig(**task_cfg_raw)
        itinerary = state_dict.get("itinerary", [])
        finalized = state_dict.get("finalized", False)
        termination_reason = state_dict.get("termination_reason", "")

        budget_score  = self._grade_budget(task_cfg, state_dict, penalties, bonuses)
        safety_score  = self._grade_safety(task_cfg, itinerary, penalties, bonuses)
        fatigue_score = self._grade_fatigue(task_cfg, itinerary, penalties, bonuses)
        cat_score     = self._grade_categories(task_cfg, state_dict, penalties, bonuses)

        fin_score = 1.0 if finalized else 0.0
        if not finalized:
            penalties.append(f"Plan not finalized (terminated via '{termination_reason}')")
        else:
            bonuses.append("Agent explicitly finalized the plan (+bonus)")

        final_score = (
            budget_score  * WEIGHT_BUDGET_COMPLIANCE +
            safety_score  * WEIGHT_SAFETY +
            fatigue_score * WEIGHT_FATIGUE +
            cat_score     * WEIGHT_CATEGORY_COVERAGE +
            fin_score     * WEIGHT_FINALIZATION
        )
        final_score = round(min(max(final_score, 0.0), 1.0), 4)

        threshold = self._PASS_THRESHOLDS.get(task_cfg.difficulty.value, 0.70)
        passed = final_score >= threshold

        summary = (
            f"Score {final_score:.4f} vs threshold {threshold} → "
            f"{'PASS' if passed else 'FAIL'}. "
            f"Cost: ${state_dict.get('total_cost', 0):.2f}/"
            f"${task_cfg.budget_usd:.2f}. "
            f"Days: {state_dict.get('days_used', 1)}/{task_cfg.total_days}."
        )

        return GradeReport(
            final_score=final_score,
            budget_score=budget_score,
            safety_score=safety_score,
            fatigue_score=fatigue_score,
            category_score=cat_score,
            finalization_score=fin_score,
            passed=passed,
            penalties=penalties,
            bonuses=bonuses,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Sub-graders
    # ------------------------------------------------------------------

    @staticmethod
    def _grade_budget(cfg: TaskConfig, state: dict, penalties: list, bonuses: list) -> float:
        spent  = state.get("total_cost", 0.0)
        budget = cfg.budget_usd

        if spent > budget:
            over_pct = (spent - budget) / budget
            penalties.append(f"Over budget by ${spent - budget:.2f} ({over_pct*100:.1f}%)")
            return max(0.0, 1.0 - over_pct * 2)

        utilisation = spent / budget if budget > 0 else 0
        if utilisation >= 0.50:
            bonuses.append(f"Good budget utilisation: {utilisation*100:.1f}% spent")
            return min(1.0, utilisation + 0.10)
        else:
            penalties.append(f"Low budget utilisation: only {utilisation*100:.1f}% spent")
            return utilisation * 1.5

    @staticmethod
    def _grade_safety(cfg: TaskConfig, itinerary: list, penalties: list, bonuses: list) -> float:
        if not itinerary:
            return 0.0
        unsafe = [item for item in itinerary if item.get("place_id") in cfg.forbidden_place_ids]
        if unsafe:
            names = ", ".join(i.get("place_name", "?") for i in unsafe)
            penalties.append(f"Forbidden place(s) visited: {names}")
            return max(0.0, 1.0 - 0.25 * len(unsafe))
        bonuses.append("All visited places are permitted and safe")
        return 1.0

    @staticmethod
    def _grade_fatigue(cfg: TaskConfig, itinerary: list, penalties: list, bonuses: list) -> float:
        max_daily = cfg.max_fatigue_index
        daily: dict[int, float] = {}
        for item in itinerary:
            d = item.get("day", 1)
            daily[d] = daily.get(d, 0.0) + item.get("fatigue_points", 0.0)

        violations = {d: f for d, f in daily.items() if f > max_daily}
        if not violations:
            bonuses.append(
                f"Fatigue well-managed (max daily: {max(daily.values(), default=0):.1f} "
                f"vs limit {max_daily})"
            )
            return 1.0

        total_excess = sum(f - max_daily for f in violations.values())
        for day, fat in violations.items():
            penalties.append(f"Day {day}: fatigue {fat:.1f} exceeds limit {max_daily}")
        penalty = min(1.0, total_excess / max_daily)
        return round(max(0.0, 1.0 - penalty), 4)

    @staticmethod
    def _grade_categories(cfg: TaskConfig, state: dict, penalties: list, bonuses: list) -> float:
        required = set(c.value if hasattr(c, 'value') else c for c in cfg.required_categories)
        covered  = set(state.get("categories_covered", []))

        if not required:
            bonuses.append("No required categories — full marks")
            return 1.0

        missing = required - covered
        if not missing:
            bonuses.append(f"All required categories covered: {', '.join(covered)}")
            return 1.0

        for cat in missing:
            penalties.append(f"Required category not visited: {cat}")
        return round(len(required - missing) / len(required), 4)
