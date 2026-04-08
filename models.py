"""
Data models for the Tour Planner Environment.

Defines Action, Observation, and State types that inherit from the
OpenEnv core base classes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    ADD_PLACE    = "add_place"
    REMOVE_PLACE = "remove_place"
    REST         = "rest"
    FINALIZE     = "finalize_plan"


class PlaceCategory(str, Enum):
    ATTRACTION = "attraction"
    RESTAURANT = "restaurant"
    HOTEL      = "hotel"
    TRANSPORT  = "transport"
    SHOPPING   = "shopping"
    NATURE     = "nature"


class DifficultyLevel(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ---------------------------------------------------------------------------
# Supporting models (plain BaseModel — NOT Action/Observation subtypes)
# ---------------------------------------------------------------------------

class Place(BaseModel):
    """A visitable location in the city catalogue."""
    place_id:        str
    name:            str
    category:        PlaceCategory
    cost:            float = Field(..., ge=0,  description="Cost in USD")
    duration_hours:  float = Field(..., gt=0,  description="Default visit duration in hours")
    fatigue_points:  float = Field(...,        description="Fatigue units (negative = recovery)")
    safety_score:    float = Field(..., ge=0, le=1, description="0 = unsafe, 1 = fully safe")
    location_tag:    str   = Field(..., description="Logical location label, e.g. 'City Centre'")
    travel_cost_from_airport: float = Field(default=0.0, ge=0)


class ItineraryItem(BaseModel):
    """One entry in the agent's planned itinerary."""
    day:            int
    place_id:       str
    place_name:     str
    duration_hours: float
    cost:           float
    fatigue_points: float
    travel_cost:    float = 0.0


class TaskConfig(BaseModel):
    """Configuration for a single task / episode."""
    task_id:             str
    difficulty:          DifficultyLevel
    total_days:          int   = Field(..., ge=1, le=14)
    budget_usd:          float = Field(..., gt=0)
    max_daily_hours:     float = Field(default=10.0, gt=0, le=24)
    max_fatigue_index:   float = Field(default=10.0, gt=0)
    required_categories: List[PlaceCategory] = Field(default_factory=list)
    forbidden_place_ids: List[str]           = Field(default_factory=list)
    starting_location:   str = "Airport"


# ---------------------------------------------------------------------------
# OpenEnv Action
# ---------------------------------------------------------------------------

class TourAction(Action):
    """
    Action for the Tour Planner environment.

    Attributes:
        type: The kind of action to execute.
        place_id: Required for add_place / remove_place.
        duration_hours: Optional override for visit duration.
    """
    type:           ActionType
    place_id:       Optional[str]   = None
    duration_hours: Optional[float] = Field(default=None, gt=0)

    @field_validator("place_id")
    @classmethod
    def place_required_for_add_remove(cls, v: Optional[str], info: Any) -> Optional[str]:
        action_type = info.data.get("type")
        if action_type in (ActionType.ADD_PLACE, ActionType.REMOVE_PLACE) and not v:
            raise ValueError(f"place_id is required for action type '{action_type}'")
        return v


# ---------------------------------------------------------------------------
# OpenEnv Observation
# ---------------------------------------------------------------------------

class TourObservation(Observation):
    """
    Observation from the Tour Planner environment.

    Contains full visibility of the current itinerary, budget, time,
    fatigue, and available places.
    """
    current_itinerary:   List[Dict[str, Any]] = Field(default_factory=list)
    remaining_budget:    float = 0.0
    days_left:           int   = 0
    current_day:         int   = 1
    hours_left_today:    float = 10.0
    current_location:    str   = "Airport"
    total_fatigue:       float = 0.0
    fatigue_warning:     bool  = False
    budget_warning:      bool  = False
    available_place_ids: List[str] = Field(default_factory=list)
    message:             Optional[str] = None


# ---------------------------------------------------------------------------
# OpenEnv State
# ---------------------------------------------------------------------------

class TourState(State, BaseModel):
    """
    Internal state for the Tour Planner environment.

    Holds the full task configuration, itinerary, and episode outcome.
    This is what the Grader consumes.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_config:        Optional[Dict[str, Any]] = None
    itinerary:          List[Dict[str, Any]] = Field(default_factory=list)
    total_cost:         float = 0.0
    total_fatigue:      float = 0.0
    days_used:          int   = 1
    categories_covered: List[str] = Field(default_factory=list)
    finalized:          bool  = False
    termination_reason: str   = ""
    city_name:          str   = "paris"
