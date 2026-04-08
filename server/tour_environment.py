"""
Tour Planner Environment - core RL environment server logic.

Implements the OpenEnv Environment interface:
    env.reset()  → TourObservation
    env.step()   → TourObservation
    env.state    → TourState  (property)
"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Optional, Set

from openenv.core.env_server.interfaces import Environment

from ..models import (
    TourAction, ActionType, TourObservation, TourState,
    Place, ItineraryItem, TaskConfig, PlaceCategory, DifficultyLevel
)

# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

REWARD_VALID_ADD       =  0.50
REWARD_CATEGORY_BONUS  =  0.30   # first time a required category is covered
REWARD_FINALIZE        =  1.00
PENALTY_DISTANCE       = -0.10   # per 10 km of travel (1 unit ≈ $10 travel cost)
PENALTY_UNSAFE         = -0.50   # safety_score < 0.90
PENALTY_FATIGUE        = -0.30   # per unit over the daily max
PENALTY_INVALID_ACTION = -0.20
PENALTY_BUDGET_BUST    = -1.00

# ---------------------------------------------------------------------------
# Built-in place catalogue (used when no city JSON is found)
# ---------------------------------------------------------------------------

_DEFAULT_PLACES: list[dict] = [
    {"place_id": "museum_01",    "name": "City History Museum",    "category": "attraction", "cost": 20,  "duration_hours": 3,   "fatigue_points": 2.0, "safety_score": 0.99, "location_tag": "City Centre",   "travel_cost_from_airport": 15},
    {"place_id": "park_01",      "name": "Riverside Nature Park",  "category": "nature",     "cost": 0,   "duration_hours": 2,   "fatigue_points": 1.0, "safety_score": 0.95, "location_tag": "Suburbs",       "travel_cost_from_airport": 25},
    {"place_id": "restaurant_01","name": "The Grand Bistro",       "category": "restaurant", "cost": 60,  "duration_hours": 1.5, "fatigue_points": 0.5, "safety_score": 0.98, "location_tag": "City Centre",   "travel_cost_from_airport": 15},
    {"place_id": "hotel_01",     "name": "Downtown Comfort Hotel", "category": "hotel",      "cost": 150, "duration_hours": 8,   "fatigue_points": -4,  "safety_score": 0.97, "location_tag": "City Centre",   "travel_cost_from_airport": 15},
    {"place_id": "gallery_01",   "name": "Modern Art Gallery",     "category": "attraction", "cost": 15,  "duration_hours": 2,   "fatigue_points": 1.5, "safety_score": 0.99, "location_tag": "Arts District", "travel_cost_from_airport": 20},
    {"place_id": "market_01",    "name": "Old Town Street Market", "category": "shopping",   "cost": 30,  "duration_hours": 2,   "fatigue_points": 1.0, "safety_score": 0.90, "location_tag": "Old Town",      "travel_cost_from_airport": 18},
    {"place_id": "temple_01",    "name": "Ancient Temple Complex", "category": "attraction", "cost": 10,  "duration_hours": 2.5, "fatigue_points": 1.5, "safety_score": 0.98, "location_tag": "Heritage Zone", "travel_cost_from_airport": 30},
    {"place_id": "restaurant_02","name": "Rooftop Fusion Kitchen", "category": "restaurant", "cost": 80,  "duration_hours": 1.5, "fatigue_points": 0.5, "safety_score": 0.97, "location_tag": "City Centre",   "travel_cost_from_airport": 15},
    {"place_id": "hotel_02",     "name": "Budget Backpacker Inn",  "category": "hotel",      "cost": 50,  "duration_hours": 8,   "fatigue_points": -2,  "safety_score": 0.88, "location_tag": "Suburbs",       "travel_cost_from_airport": 25},
    {"place_id": "beach_01",     "name": "Sunrise Beach",          "category": "nature",     "cost": 5,   "duration_hours": 3,   "fatigue_points": 1.0, "safety_score": 0.93, "location_tag": "Coastal",       "travel_cost_from_airport": 40},
    {"place_id": "zoo_01",       "name": "National Wildlife Zoo",  "category": "attraction", "cost": 25,  "duration_hours": 4,   "fatigue_points": 2.5, "safety_score": 0.99, "location_tag": "Suburbs",       "travel_cost_from_airport": 22},
    {"place_id": "spa_01",       "name": "Serenity Wellness Spa",  "category": "attraction", "cost": 120, "duration_hours": 3,   "fatigue_points": -3,  "safety_score": 0.99, "location_tag": "City Centre",   "travel_cost_from_airport": 15},
]

# ---------------------------------------------------------------------------
# Default task configurations
# ---------------------------------------------------------------------------

_TASK_CONFIGS: dict[str, dict] = {
    "task_1_easy": {
        "task_id": "task_1_easy", "difficulty": "easy",
        "total_days": 2, "budget_usd": 500,
        "max_daily_hours": 10, "max_fatigue_index": 12,
        "required_categories": ["attraction", "restaurant"],
        "forbidden_place_ids": [], "starting_location": "Airport",
    },
    "task_2_medium": {
        "task_id": "task_2_medium", "difficulty": "medium",
        "total_days": 3, "budget_usd": 1000,
        "max_daily_hours": 10, "max_fatigue_index": 10,
        "required_categories": ["attraction", "restaurant", "nature"],
        "forbidden_place_ids": [], "starting_location": "Airport",
    },
    "task_3_hard": {
        "task_id": "task_3_hard", "difficulty": "hard",
        "total_days": 5, "budget_usd": 1500,
        "max_daily_hours": 10, "max_fatigue_index": 8,
        "required_categories": ["attraction", "restaurant", "nature", "shopping"],
        "forbidden_place_ids": ["hotel_02"],
        "starting_location": "Airport",
    },
}


class TourPlannerEnvironment(Environment[TourAction, TourObservation, TourState]):
    """
    OpenEnv-compatible tour-planning RL environment.

    The agent must build a multi-day itinerary by adding/removing places,
    resting to advance days, and finalizing a plan - all within budget
    and fatigue constraints.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._task_cfg: Optional[TaskConfig] = None
        self._catalogue: dict[str, Place] = {}
        self._itinerary: list[ItineraryItem] = []
        self._remaining_budget: float = 0.0
        self._current_day: int = 1
        self._hours_left_today: float = 10.0
        self._current_location: str = "Airport"
        self._total_fatigue: float = 0.0
        self._total_cost: float = 0.0
        self._finalized: bool = False
        self._done: bool = False
        self._termination_reason: str = ""
        self._covered_categories: set[PlaceCategory] = set()
        self._city_name: str = "default"
        self._episode_id: Optional[str] = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "task_2_medium",
        city_name: str = "default",
        **kwargs: Any,
    ) -> TourObservation:
        """Reset to a fresh episode and return the initial observation."""
        self._reset_rubric()

        # Load task config
        raw_cfg = _TASK_CONFIGS.get(task_id, _TASK_CONFIGS["task_2_medium"])
        self._task_cfg = TaskConfig(**raw_cfg)

        # Load city catalogue
        self._city_name = city_name
        self._catalogue = self._load_catalogue(city_name)

        # Reset episode state
        self._episode_id = episode_id or str(uuid.uuid4())
        self._itinerary = []
        self._remaining_budget = self._task_cfg.budget_usd
        self._current_day = 1
        self._hours_left_today = self._task_cfg.max_daily_hours
        self._current_location = self._task_cfg.starting_location
        self._total_fatigue = 0.0
        self._total_cost = 0.0
        self._finalized = False
        self._done = False
        self._termination_reason = ""
        self._covered_categories = set()

        return self._make_observation()

    def step(
        self,
        action: TourAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TourObservation:
        """Apply action and advance the environment by one step."""
        if self._done:
            obs = self._make_observation()
            obs.done = True
            obs.reward = 0.0
            obs.message = "Episode already finished. Call reset()."
            return obs

        reward, info = self._dispatch(action)
        self._check_terminal_conditions()

        obs = self._make_observation()
        obs.reward = reward
        obs.done = self._done
        obs.metadata = info
        return obs

    @property
    def state(self) -> TourState:
        """Return a snapshot of the complete episode state (used by Grader)."""
        covered_cats = list({
            self._catalogue[item.place_id].category.value
            for item in self._itinerary
            if item.place_id in self._catalogue
        })

        return TourState(
            episode_id=self._episode_id,
            step_count=len(self._itinerary),
            task_config=self._task_cfg.model_dump() if self._task_cfg else None,
            itinerary=[item.model_dump() for item in self._itinerary],
            total_cost=self._total_cost,
            total_fatigue=self._total_fatigue,
            days_used=self._current_day,
            categories_covered=covered_cats,
            finalized=self._finalized,
            termination_reason=self._termination_reason,
            city_name=self._city_name,
        )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _dispatch(self, action: TourAction) -> tuple[float, dict]:
        if action.type == ActionType.ADD_PLACE:
            return self._handle_add(action)
        if action.type == ActionType.REMOVE_PLACE:
            return self._handle_remove(action)
        if action.type == ActionType.REST:
            return self._handle_rest()
        if action.type == ActionType.FINALIZE:
            return self._handle_finalize()
        return PENALTY_INVALID_ACTION, {"error": "Unknown action type"}

    def _handle_add(self, action: TourAction) -> tuple[float, dict]:
        place_id = action.place_id
        if place_id not in self._catalogue:
            return PENALTY_INVALID_ACTION, {"error": f"Unknown place_id: {place_id}"}

        place = self._catalogue[place_id]
        duration = action.duration_hours or place.duration_hours

        # Forbidden check
        if place_id in self._task_cfg.forbidden_place_ids:
            return PENALTY_INVALID_ACTION, {"error": f"{place_id} is forbidden in this task"}

        # Budget check
        travel_cost = self._travel_cost(place)
        total_item_cost = place.cost + travel_cost
        if total_item_cost > self._remaining_budget:
            self._done = True
            self._termination_reason = "budget_exceeded"
            return PENALTY_BUDGET_BUST, {"error": "Budget exceeded", "cost": total_item_cost}

        # Daily hours check - auto-advance day if needed
        if duration > self._hours_left_today:
            if self._current_day >= self._task_cfg.total_days:
                self._done = True
                self._termination_reason = "time_exceeded"
                return PENALTY_INVALID_ACTION, {"error": "No days left"}
            self._current_day += 1
            self._hours_left_today = self._task_cfg.max_daily_hours

        # Safety check
        reward = REWARD_VALID_ADD
        if place.safety_score < 0.90:
            reward += PENALTY_UNSAFE

        # Travel distance penalty
        reward += PENALTY_DISTANCE * (travel_cost / 10)

        # Category bonus
        if place.category in self._task_cfg.required_categories and \
                place.category not in self._covered_categories:
            reward += REWARD_CATEGORY_BONUS
            self._covered_categories.add(place.category)

        # Commit
        item = ItineraryItem(
            day=self._current_day,
            place_id=place_id,
            place_name=place.name,
            duration_hours=duration,
            cost=place.cost,
            fatigue_points=place.fatigue_points,
            travel_cost=travel_cost,
        )
        self._itinerary.append(item)
        self._remaining_budget -= total_item_cost
        self._total_cost       += total_item_cost
        self._hours_left_today -= duration
        self._total_fatigue     = max(0, self._total_fatigue + place.fatigue_points)
        self._current_location  = place.location_tag

        # Fatigue penalty
        day_fatigue = sum(
            i.fatigue_points for i in self._itinerary if i.day == self._current_day
        )
        if day_fatigue > self._task_cfg.max_fatigue_index:
            excess = day_fatigue - self._task_cfg.max_fatigue_index
            reward += PENALTY_FATIGUE * excess

        return round(reward, 4), {
            "added": place.name,
            "cost": total_item_cost,
            "day": self._current_day,
            "fatigue_warning": day_fatigue > self._task_cfg.max_fatigue_index * 0.8,
        }

    def _handle_remove(self, action: TourAction) -> tuple[float, dict]:
        place_id = action.place_id
        for i, item in enumerate(self._itinerary):
            if item.place_id == place_id:
                self._itinerary.pop(i)
                self._remaining_budget += item.cost + item.travel_cost
                self._total_cost       -= item.cost + item.travel_cost
                self._total_fatigue     = max(0, self._total_fatigue - item.fatigue_points)
                return 0.0, {"removed": item.place_name}
        return PENALTY_INVALID_ACTION, {"error": f"{place_id} not in itinerary"}

    def _handle_rest(self) -> tuple[float, dict]:
        if self._current_day >= self._task_cfg.total_days:
            self._done = True
            self._termination_reason = "time_exceeded"
            return 0.0, {"message": "No more days to rest into"}
        self._current_day      += 1
        self._hours_left_today  = self._task_cfg.max_daily_hours
        self._total_fatigue     = max(0, self._total_fatigue - 2)
        return 0.10, {"message": f"Rested. Now on day {self._current_day}"}

    def _handle_finalize(self) -> tuple[float, dict]:
        self._finalized          = True
        self._done               = True
        self._termination_reason = "finalized"
        return REWARD_FINALIZE, {"message": "Plan finalized. Episode complete."}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_terminal_conditions(self) -> None:
        if self._remaining_budget <= 0:
            self._done = True
            self._termination_reason = "budget_exceeded"

    def _travel_cost(self, place: Place) -> float:
        visited_locations = {i.place_id for i in self._itinerary}
        if not visited_locations or self._current_location == "Airport":
            return place.travel_cost_from_airport
        if self._current_location == place.location_tag:
            return 0.0
        return round(place.travel_cost_from_airport * 0.5, 2)

    def _make_observation(self) -> TourObservation:
        day_fatigue = sum(
            i.fatigue_points for i in self._itinerary if i.day == self._current_day
        )
        return TourObservation(
            current_itinerary=[item.model_dump() for item in self._itinerary],
            remaining_budget=round(self._remaining_budget, 2),
            days_left=max(0, self._task_cfg.total_days - self._current_day),
            current_day=self._current_day,
            hours_left_today=round(self._hours_left_today, 2),
            current_location=self._current_location,
            total_fatigue=round(self._total_fatigue, 2),
            fatigue_warning=day_fatigue > self._task_cfg.max_fatigue_index * 0.8,
            budget_warning=self._remaining_budget < self._task_cfg.budget_usd * 0.15,
            available_place_ids=list(self._catalogue.keys()),
        )

    def _load_catalogue(self, city_name: str) -> dict[str, Place]:
        """Try to load a city JSON from catalogues/, fall back to defaults."""
        catalogue_dir = Path(__file__).parent.parent / "catalogues"
        city_file = catalogue_dir / f"{city_name.lower()}.json"
        if city_file.exists():
            raw = json.loads(city_file.read_text())
        else:
            raw = _DEFAULT_PLACES
        return {p["place_id"]: Place(**p) for p in raw}
