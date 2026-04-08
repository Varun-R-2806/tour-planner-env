"""
TourPlannerEnv Client.

Provides the client for connecting to a Tour Planner Environment server
via WebSocket for persistent sessions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import TourAction, TourObservation, TourState

if TYPE_CHECKING:
    from openenv.core.containers.runtime import ContainerProvider


class TourPlannerEnv(EnvClient[TourAction, TourObservation, TourState]):
    """
    Client for Tour Planner Environment.

    Maintains a persistent WebSocket connection to the environment
    server, enabling multi-step interactions.

    Example:
        >>> with TourPlannerEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.remaining_budget)
        ...
        ...     result = client.step(TourAction(type="add_place", place_id="museum_01"))
        ...     print(result.observation.reward)

    Example with Docker:
        >>> client = TourPlannerEnv.from_docker_image("tour-planner-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(TourAction(type="add_place", place_id="park_01"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: TourAction) -> Dict[str, Any]:
        """Convert TourAction to JSON payload for step request."""
        payload = {"type": action.type.value}
        if action.place_id is not None:
            payload["place_id"] = action.place_id
        if action.duration_hours is not None:
            payload["duration_hours"] = action.duration_hours
        return payload

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[TourObservation]:
        """Parse server response into StepResult[TourObservation]."""
        obs_data = payload.get("observation", {})

        observation = TourObservation(
            current_itinerary=obs_data.get("current_itinerary", []),
            remaining_budget=obs_data.get("remaining_budget", 0.0),
            days_left=obs_data.get("days_left", 0),
            current_day=obs_data.get("current_day", 1),
            hours_left_today=obs_data.get("hours_left_today", 10.0),
            current_location=obs_data.get("current_location", "Airport"),
            total_fatigue=obs_data.get("total_fatigue", 0.0),
            fatigue_warning=obs_data.get("fatigue_warning", False),
            budget_warning=obs_data.get("budget_warning", False),
            available_place_ids=obs_data.get("available_place_ids", []),
            message=obs_data.get("message"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TourState:
        """Parse server response into TourState."""
        return TourState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_config=payload.get("task_config"),
            itinerary=payload.get("itinerary", []),
            total_cost=payload.get("total_cost", 0.0),
            total_fatigue=payload.get("total_fatigue", 0.0),
            days_used=payload.get("days_used", 1),
            categories_covered=payload.get("categories_covered", []),
            finalized=payload.get("finalized", False),
            termination_reason=payload.get("termination_reason", ""),
            city_name=payload.get("city_name", "default"),
        )
