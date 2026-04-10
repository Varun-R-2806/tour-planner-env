"""
Data models for the Tour Planner Environment V2.

Defines Action, Observation, and State types that inherit from the
OpenEnv core base classes, with advanced dynamic environments.
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
    SEARCH_PLACES     = "search_places"
    GET_PLACE_DETAILS = "get_place_details"
    BOOK_TICKET       = "book_ticket"
    WAIT              = "wait"
    FINALIZE          = "finalize_plan"

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
    """A visitable location in the city catalogue with extended real-world properties."""
    place_id:        str
    name:            str
    category:        PlaceCategory
    cost:            float = Field(..., ge=0,  description="Cost in USD")
    duration_hours:  float = Field(..., gt=0,  description="Default visit duration in hours")
    fatigue_points:  float = Field(...,        description="Fatigue units (negative = recovery)")
    safety_score:    float = Field(..., ge=0, le=1, description="0 = unsafe, 1 = fully safe")
    location_tag:    str   = Field(..., description="Logical location label, e.g. 'City Centre'")
    travel_cost_from_airport: float = Field(default=0.0, ge=0)
    
    # New V2 Properties
    opening_time:    str = Field(default="09:00", description="HH:MM format")
    closing_time:    str = Field(default="20:00", description="HH:MM format")
    is_outdoor:      bool = Field(default=False, description="Whether weather affects this place")
    requires_booking: bool = Field(default=False, description="Must be booked in advance")


class ItineraryItem(BaseModel):
    """One entry in the agent's planned itinerary."""
    day:            int
    start_time:     str   = "00:00"
    end_time:       str   = "00:00"
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
    max_daily_hours:     float = Field(default=12.0, gt=0, le=24)
    max_fatigue_index:   float = Field(default=10.0, gt=0)
    required_categories: List[PlaceCategory] = Field(default_factory=list)
    forbidden_place_ids: List[str]           = Field(default_factory=list)
    starting_location:   str = "Airport"
    
    # New V2 Persona rules
    user_persona:        str = "A generic tourist."


# ---------------------------------------------------------------------------
# OpenEnv Action
# ---------------------------------------------------------------------------

class TourAction(Action):
    """
    Action for the Tour Planner environment V2.

    Attributes:
        type: The kind of action to execute.
        place_id: Required for get_place_details / book_ticket.
        query: Optional search keyword.
        category: Optional search category filter.
        hours_to_wait: Used for WAIT action to advance clock.
    """
    type:           ActionType
    place_id:       Optional[str]   = None
    query:          Optional[str]   = None
    category:       Optional[PlaceCategory] = None
    hours_to_wait:  Optional[float] = Field(default=None, gt=0)
    
    @field_validator("place_id")
    @classmethod
    def require_place_id(cls, v: Optional[str], info: Any) -> Optional[str]:
        action_type = info.data.get("type")
        if action_type in (ActionType.GET_PLACE_DETAILS, ActionType.BOOK_TICKET) and not v:
            raise ValueError(f"place_id is required for action type '{action_type}'")
        return v


# ---------------------------------------------------------------------------
# OpenEnv Observation
# ---------------------------------------------------------------------------

class TourObservation(Observation):
    """
    Observation from the Tour Planner environment V2.

    The AI is blind to the catalogue and must rely on search_results.
    """
    current_itinerary:   List[Dict[str, Any]] = Field(default_factory=list)
    remaining_budget:    float = 0.0
    days_left:           int   = 0
    current_day:         int   = 1
    current_time:        str   = "08:00"  # HH:MM representation
    current_location:    str   = "Airport"
    total_fatigue:       float = 0.0
    fatigue_warning:     bool  = False
    budget_warning:      bool  = False
    
    # Context injected by action results instead of full catalogue dump
    search_results:      Optional[List[Dict[str, Any]]] = None
    place_details:       Optional[Dict[str, Any]] = None
    message:             Optional[str] = None
    
    # V2 Dynamic Events
    current_weather:     str = "Sunny"


# ---------------------------------------------------------------------------
# OpenEnv State
# ---------------------------------------------------------------------------

class TourState(State, BaseModel):
    """Internal state for the Grader."""
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
    
    # Hidden dynamic events tracked for grading
    weather_events:     Dict[int, str] = Field(default_factory=dict)
