"""
FastAPI application for the Tour Planner Environment.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Or run directly:
    uv run --project . server
"""

import os

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.http_server import create_app

    from ..models import TourAction, TourObservation
    from .tour_environment import TourPlannerEnvironment
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from envs.tour_planner_env.models import TourAction, TourObservation
    from openenv.core.env_server.http_server import create_app
    from envs.tour_planner_env.server.tour_environment import TourPlannerEnvironment


# Get configuration from environment variables
default_task = os.getenv("TOUR_PLANNER_TASK", "task_2_medium")
default_city = os.getenv("TOUR_PLANNER_CITY", "default")


def create_tour_planner_environment():
    """Factory function that creates TourPlannerEnvironment instances."""
    return TourPlannerEnvironment()


# Create the FastAPI app
app = create_app(
    create_tour_planner_environment,
    TourAction,
    TourObservation,
    env_name="tour_planner_env",
)

# ---------------------------------------------------------------------------
# /tasks endpoint — required by the hackathon validator to enumerate tasks
# ---------------------------------------------------------------------------

from fastapi import FastAPI
from fastapi.responses import JSONResponse

TASKS = [
    {
        "task_id": "task_1_easy",
        "difficulty": "easy",
        "description": "Plan a 2-day tour within a $500 budget covering attractions and restaurants.",
        "has_grader": True,
        "score_range": [0.0, 1.0],
    },
    {
        "task_id": "task_2_medium",
        "difficulty": "medium",
        "description": "Plan a 3-day tour within a $1000 budget covering attractions, restaurants, and nature.",
        "has_grader": True,
        "score_range": [0.0, 1.0],
    },
    {
        "task_id": "task_3_hard",
        "difficulty": "hard",
        "description": "Plan a 5-day tour within a $1500 budget covering 4 categories with fatigue and safety constraints.",
        "has_grader": True,
        "score_range": [0.0, 1.0],
    },
]

@app.get("/tasks")
def list_tasks():
    """Return all available tasks with grader metadata."""
    return JSONResponse(content={"tasks": TASKS})


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m tour_planner_env.server.app
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
