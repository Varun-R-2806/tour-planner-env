"""
Tour Planner Environment - core RL environment server logic (V2).

Implements the OpenEnv Environment interface:
    env.reset()  → TourObservation
    env.step()   → TourObservation
    env.state    → TourState  (property)

V2 Upgrade: Introduces strict temporal booking, dynamic weather events,
and hidden place catalogues requiring search tools.
"""

from __future__ import annotations

import json
import math
import random
import uuid
import os
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
REWARD_CATEGORY_BONUS  =  0.30
REWARD_FINALIZE        =  1.00
PENALTY_DISTANCE       = -0.10
PENALTY_UNSAFE         = -0.50
PENALTY_FATIGUE        = -0.30
PENALTY_INVALID_ACTION = -0.20
PENALTY_BUDGET_BUST    = -1.00

# ---------------------------------------------------------------------------
# Built-in place catalogue
# ---------------------------------------------------------------------------

_DEFAULT_PLACES: list[dict] = [
    {"place_id": "museum_01",    "name": "City History Museum",    "category": "attraction", "cost": 20,  "duration_hours": 3,   "fatigue_points": 2.0, "safety_score": 0.99, "location_tag": "City Centre",   "travel_cost_from_airport": 15, "opening_time": "09:00", "closing_time": "17:00", "is_outdoor": False, "requires_booking": True},
    {"place_id": "park_01",      "name": "Riverside Nature Park",  "category": "nature",     "cost": 0,   "duration_hours": 2,   "fatigue_points": 1.0, "safety_score": 0.95, "location_tag": "Suburbs",       "travel_cost_from_airport": 25, "opening_time": "06:00", "closing_time": "19:00", "is_outdoor": True, "requires_booking": False},
    {"place_id": "restaurant_01","name": "The Grand Bistro",       "category": "restaurant", "cost": 60,  "duration_hours": 1.5, "fatigue_points": 0.5, "safety_score": 0.98, "location_tag": "City Centre",   "travel_cost_from_airport": 15, "opening_time": "11:00", "closing_time": "23:00", "is_outdoor": False, "requires_booking": True},
    {"place_id": "hotel_01",     "name": "Downtown Comfort Hotel", "category": "hotel",      "cost": 150, "duration_hours": 8,   "fatigue_points": -4,  "safety_score": 0.97, "location_tag": "City Centre",   "travel_cost_from_airport": 15, "opening_time": "00:00", "closing_time": "23:59", "is_outdoor": False, "requires_booking": True},
    {"place_id": "gallery_01",   "name": "Modern Art Gallery",     "category": "attraction", "cost": 15,  "duration_hours": 2,   "fatigue_points": 1.5, "safety_score": 0.99, "location_tag": "Arts District", "travel_cost_from_airport": 20, "opening_time": "10:00", "closing_time": "18:00", "is_outdoor": False, "requires_booking": False},
    {"place_id": "market_01",    "name": "Old Town Street Market", "category": "shopping",   "cost": 30,  "duration_hours": 2,   "fatigue_points": 1.0, "safety_score": 0.90, "location_tag": "Old Town",      "travel_cost_from_airport": 18, "opening_time": "08:00", "closing_time": "16:00", "is_outdoor": True, "requires_booking": False},
    {"place_id": "temple_01",    "name": "Ancient Temple Complex", "category": "attraction", "cost": 10,  "duration_hours": 2.5, "fatigue_points": 1.5, "safety_score": 0.98, "location_tag": "Heritage Zone", "travel_cost_from_airport": 30, "opening_time": "07:00", "closing_time": "20:00", "is_outdoor": True, "requires_booking": False},
    {"place_id": "restaurant_02","name": "Rooftop Fusion Kitchen", "category": "restaurant", "cost": 80,  "duration_hours": 1.5, "fatigue_points": 0.5, "safety_score": 0.97, "location_tag": "City Centre",   "travel_cost_from_airport": 15, "opening_time": "17:00", "closing_time": "02:00", "is_outdoor": True, "requires_booking": True},
    {"place_id": "hotel_02",     "name": "Budget Backpacker Inn",  "category": "hotel",      "cost": 50,  "duration_hours": 8,   "fatigue_points": -2,  "safety_score": 0.88, "location_tag": "Suburbs",       "travel_cost_from_airport": 25, "opening_time": "00:00", "closing_time": "23:59", "is_outdoor": False, "requires_booking": True},
    {"place_id": "beach_01",     "name": "Sunrise Beach",          "category": "nature",     "cost": 5,   "duration_hours": 3,   "fatigue_points": 1.0, "safety_score": 0.93, "location_tag": "Coastal",       "travel_cost_from_airport": 40, "opening_time": "05:00", "closing_time": "21:00", "is_outdoor": True, "requires_booking": False},
    {"place_id": "zoo_01",       "name": "National Wildlife Zoo",  "category": "attraction", "cost": 25,  "duration_hours": 4,   "fatigue_points": 2.5, "safety_score": 0.99, "location_tag": "Suburbs",       "travel_cost_from_airport": 22, "opening_time": "08:30", "closing_time": "16:30", "is_outdoor": True, "requires_booking": True},
    {"place_id": "spa_01",       "name": "Serenity Wellness Spa",  "category": "attraction", "cost": 120, "duration_hours": 3,   "fatigue_points": -3,  "safety_score": 0.99, "location_tag": "City Centre",   "travel_cost_from_airport": 15, "opening_time": "09:00", "closing_time": "21:00", "is_outdoor": False, "requires_booking": True},
]

# ---------------------------------------------------------------------------
# Default task configurations
# ---------------------------------------------------------------------------

_TASK_CONFIGS: dict[str, dict] = {
    "task_1_easy": {
        "task_id": "task_1_easy", "difficulty": "easy",
        "total_days": 2, "budget_usd": 500,
        "max_daily_hours": 12, "max_fatigue_index": 12,
        "required_categories": ["attraction", "restaurant"],
        "forbidden_place_ids": [], "starting_location": "Airport",
        "user_persona": "A first-time tourist looking for standard attractions. Does not care about weather."
    },
    "task_2_medium": {
        "task_id": "task_2_medium", "difficulty": "medium",
        "total_days": 3, "budget_usd": 1000,
        "max_daily_hours": 12, "max_fatigue_index": 10,
        "required_categories": ["attraction", "restaurant", "nature"],
        "forbidden_place_ids": [], "starting_location": "Airport",
        "user_persona": "An active traveler wanting a balanced mix of nature and dining."
    },
    "task_3_hard": {
        "task_id": "task_3_hard", "difficulty": "hard",
        "total_days": 5, "budget_usd": 1500,
        "max_daily_hours": 10, "max_fatigue_index": 8,
        "required_categories": ["attraction", "restaurant", "nature", "shopping"],
        "forbidden_place_ids": ["hotel_02"],
        "starting_location": "Airport",
        "user_persona": "A disabled traveler who requires low fatigue, highly safe venues, and hates being outdoors in the rain."
    },
}

def time_to_float(time_str: str) -> float:
    h, m = map(int, time_str.split(':'))
    return h + m / 60.0

def float_to_time(time_float: float) -> str:
    h = int(time_float)
    m = int(round((time_float - h) * 60))
    if m == 60:
        h += 1
        m = 0
    return f"{h:02d}:{m:02d}"

class TourPlannerEnvironment(Environment[TourAction, TourObservation, TourState]):
    """OpenEnv-compatible tour-planning RL environment V2."""

    def __init__(self, **kwargs):
        super().__init__()
        self._task_cfg: Optional[TaskConfig] = None
        self._catalogue: dict[str, Place] = {}
        self._itinerary: list[ItineraryItem] = []
        self._remaining_budget: float = 0.0
        self._current_day: int = 1
        self._current_time: float = 8.0 # Starts at 08:00
        self._current_location: str = "Airport"
        self._total_fatigue: float = 0.0
        self._total_cost: float = 0.0
        self._finalized: bool = False
        self._done: bool = False
        self._termination_reason: str = ""
        self._covered_categories: set[PlaceCategory] = set()
        self._city_name: str = "default"
        self._episode_id: Optional[str] = None
        self._weather_events: dict[int, str] = {}
        
        # Temp action states
        self._last_search_results: list = []
        self._last_place_details: dict = {}
        self._last_message: str = ""

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TourObservation:
        self._reset_rubric()
        target_task = os.getenv("TASK_NAME", "task_2_medium")
        options = kwargs.get("options", {})
        if options and isinstance(options, dict) and "task_name" in options:
            target_task = options["task_name"]
        elif "task_name" in kwargs:
            target_task = kwargs["task_name"]
        elif "task_id" in kwargs:
            target_task = kwargs["task_id"]

        city_name = kwargs.get("city_name", options.get("city_name", "default"))
        if target_task not in _TASK_CONFIGS:
            target_task = "task_2_medium"

        raw_cfg = _TASK_CONFIGS[target_task]
        self._task_cfg = TaskConfig(**raw_cfg)
        self._city_name = city_name
        self._catalogue = self._load_catalogue(city_name)
        self._episode_id = episode_id or str(uuid.uuid4())
        
        # Generate weather events
        if seed is not None:
            random.seed(seed)
        weathers = ["Sunny", "Cloudy", "Light Rain", "Heavy Rain", "Thunderstorm"]
        self._weather_events = {day: random.choice(weathers) for day in range(1, self._task_cfg.total_days + 1)}

        self._itinerary = []
        self._remaining_budget = self._task_cfg.budget_usd
        self._current_day = 1
        self._current_time = 8.0
        self._current_location = self._task_cfg.starting_location
        self._total_fatigue = 0.0
        self._total_cost = 0.0
        self._finalized = False
        self._done = False
        self._termination_reason = ""
        self._covered_categories = set()
        self._last_search_results = []
        self._last_place_details = {}
        self._last_message = "Welcome to Tour Planner V2. Use search_places to begin."

        return self._make_observation()

    def step(
        self,
        action: TourAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TourObservation:
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
            weather_events=self._weather_events,
        )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _dispatch(self, action: TourAction) -> tuple[float, dict]:
        self._last_search_results = None
        self._last_place_details = None
        self._last_message = ""
        
        if action.type == ActionType.SEARCH_PLACES:
            return self._handle_search(action)
        if action.type == ActionType.GET_PLACE_DETAILS:
            return self._handle_details(action)
        if action.type == ActionType.BOOK_TICKET:
            return self._handle_book(action)
        if action.type == ActionType.WAIT:
            return self._handle_wait(action)
        if action.type == ActionType.FINALIZE:
            return self._handle_finalize()
        return PENALTY_INVALID_ACTION, {"error": "Unknown action type"}

    def _handle_search(self, action: TourAction) -> tuple[float, dict]:
        results = []
        for pid, p in self._catalogue.items():
            if action.category and p.category != action.category:
                continue
            if action.query and action.query.lower() not in p.name.lower() and action.query.lower() not in p.location_tag.lower():
                continue
            results.append({"place_id": p.place_id, "name": p.name, "category": p.category.value})
        
        self._last_search_results = results
        self._last_message = f"Found {len(results)} places matching criteria."
        return 0.1, {"results_count": len(results)}

    def _handle_details(self, action: TourAction) -> tuple[float, dict]:
        place_id = action.place_id
        if place_id not in self._catalogue:
            self._last_message = f"Error: Unknown place_id '{place_id}'"
            return PENALTY_INVALID_ACTION, {"error": "Unknown place"}
        
        p = self._catalogue[place_id]
        self._last_place_details = p.model_dump()
        self._last_message = f"Showing details for {p.name}."
        return 0.1, {"place": p.name}

    def _handle_wait(self, action: TourAction) -> tuple[float, dict]:
        hours = action.hours_to_wait or 1.0
        self._current_time += hours
        if self._current_time >= 24.0:
            self._current_time -= 24.0
            self._current_day += 1
            self._total_fatigue = max(0, self._total_fatigue - 2.0)
            
            if self._current_day > self._task_cfg.total_days:
                self._done = True
                self._termination_reason = "time_exceeded"
                self._last_message = "Ran out of vacation days!"
                return PENALTY_INVALID_ACTION, {"error": "No days left"}
                
        self._last_message = f"Waited for {hours} hours. Time is now {float_to_time(self._current_time)}."
        return 0.05, {"waited": hours}

    def _handle_book(self, action: TourAction) -> tuple[float, dict]:
        place_id = action.place_id
        if place_id not in self._catalogue:
            self._last_message = f"Error: Unknown place_id '{place_id}'"
            return PENALTY_INVALID_ACTION, {"error": "Unknown place"}

        place = self._catalogue[place_id]
        
        if place_id in self._task_cfg.forbidden_place_ids:
            self._last_message = f"Error: {place.name} is forbidden based on persona constraints."
            return PENALTY_INVALID_ACTION, {"error": "Forbidden place"}

        # Commute time logic
        travel_cost = self._travel_cost(place)
        travel_time_hours = travel_cost / 10.0 # roughly 10 dollars = 1 hour travel
        arrival_time = self._current_time + travel_time_hours
        
        # Opening logic
        open_fl = time_to_float(place.opening_time)
        close_fl = time_to_float(place.closing_time)
        duration = place.duration_hours
        
        if arrival_time < open_fl or (arrival_time + duration) > close_fl:
            self._last_message = f"Error: The venue '{place.name}' is closed. Operating hours: {place.opening_time}-{place.closing_time}. You would arrive at {float_to_time(arrival_time)}."
            return PENALTY_INVALID_ACTION, {"error": "Closed"}

        # Budget logic
        total_cost = place.cost + travel_cost
        if total_cost > self._remaining_budget:
            self._done = True
            self._termination_reason = "budget_exceeded"
            self._last_message = f"Error: Budget exceeded trying to book {place.name}."
            return PENALTY_BUDGET_BUST, {"error": "Budget busted"}

        # Success! Book the ticket
        item = ItineraryItem(
            day=self._current_day,
            start_time=float_to_time(arrival_time),
            end_time=float_to_time(arrival_time + duration),
            place_id=place_id,
            place_name=place.name,
            duration_hours=duration,
            cost=place.cost,
            fatigue_points=place.fatigue_points,
            travel_cost=travel_cost,
        )
        self._itinerary.append(item)
        
        self._remaining_budget -= total_cost
        self._total_cost += total_cost
        self._total_fatigue = max(0, self._total_fatigue + place.fatigue_points)
        self._current_location = place.location_tag
        self._current_time = arrival_time + duration
        
        reward = REWARD_VALID_ADD
        if place.safety_score < 0.90:
            reward += PENALTY_UNSAFE
        reward += PENALTY_DISTANCE * travel_time_hours
        if place.category in self._task_cfg.required_categories and place.category not in self._covered_categories:
            reward += REWARD_CATEGORY_BONUS
            self._covered_categories.add(place.category)
            
        self._last_message = f"Successfully booked {place.name}! Current time is now {float_to_time(self._current_time)}."
        return round(reward, 4), {"added": place.name, "cost": total_cost}

    def _handle_finalize(self) -> tuple[float, dict]:
        self._finalized = True
        self._done = True
        self._termination_reason = "finalized"
        self._last_message = "Tour planner finalized. Awaiting Grader execution."
        return REWARD_FINALIZE, {"message": "Plan finalized."}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_terminal_conditions(self) -> None:
        if self._remaining_budget <= 0:
            self._done = True
            self._termination_reason = "budget_exceeded"

    def _travel_cost(self, place: Place) -> float:
        if self._current_location == "Airport" or not self._itinerary:
            return place.travel_cost_from_airport
        if self._current_location == place.location_tag:
            return 0.0
        return round(place.travel_cost_from_airport * 0.5, 2)

    def _make_observation(self) -> TourObservation:
        day_fatigue = sum(i.fatigue_points for i in self._itinerary if i.day == self._current_day)
        return TourObservation(
            current_itinerary=[item.model_dump() for item in self._itinerary],
            remaining_budget=round(self._remaining_budget, 2),
            days_left=max(0, self._task_cfg.total_days - self._current_day),
            current_day=self._current_day,
            current_time=float_to_time(self._current_time),
            current_location=self._current_location,
            total_fatigue=round(self._total_fatigue, 2),
            fatigue_warning=day_fatigue > self._task_cfg.max_fatigue_index * 0.8,
            budget_warning=self._remaining_budget < self._task_cfg.budget_usd * 0.15,
            search_results=self._last_search_results,
            place_details=self._last_place_details,
            message=self._last_message,
            current_weather=self._weather_events.get(self._current_day, "Sunny"),
        )

    def _load_catalogue(self, city_name: str) -> dict[str, Place]:
        catalogue_dir = Path(__file__).parent.parent / "catalogues"
        city_file = catalogue_dir / f"{city_name.lower()}.json"
        if city_file.exists():
            raw = json.loads(city_file.read_text())
        else:
            raw = _DEFAULT_PLACES
            
        validated_places = {}
        for p in raw:
            # Inject defaults for older JSON templates compatible with v1
            if "opening_time" not in p: p["opening_time"] = "09:00"
            if "closing_time" not in p: p["closing_time"] = "20:00"
            if "is_outdoor" not in p: p["is_outdoor"] = False
            if "requires_booking" not in p: p["requires_booking"] = False
            validated_places[p["place_id"]] = Place(**p)
            
        return validated_places
