# Tour Planner RL Environment

The Tour Planner environment is a multi-step, multi-day simulation designed to test agents' ability to plan itineraries given real-world constraints:
- **Budget**: USD cap per episode.
- **Fatigue**: Agents must manage daily fatigue by choosing to **rest**.
- **Safety**: Some places are risky.
- **Categories**: Specific tasks may require visits to different categories (nature, attraction, etc.).

## Cities
- **Paris**: Art, history, and fine dining.
- **Tokyo**: Tech, temples, and luxury.
- **London**: Museums and heritage.
- **Mumbai**: Culture, food, and history.
- **New York**: Shopping and landmarks.

## Features
- **Deterministic Grader**: Programs scoring from 0.0 to 1.0 based on compliance and efficiency.
- **OpenEnv Specification**: Fully compliant with the OpenEnv interface.
- **Async Support**: WebSocket-ready for fast inference.

## Testing Locally

### 1. Start Server
```powershell
$env:PYTHONPATH = "src;envs"
openenv serve envs/tour_planner_env
# (Alternatively, run uvicorn manually:)
# python -m tour_planner_env.server.app
```

### 2. Run Baseline Test (No LLM)
```powershell
python test_locally.py
```

### 3. Run Agent (LLM Required)
```powershell
$env:API_BASE_URL = "..." # defaults to OpenAI
$env:MODEL_NAME = "gpt-4o"
$env:OPENAI_API_KEY = "..."
python inference.py --task task_2_medium --city tokyo
```

### 4. Build Docker
```powershell
openenv build envs/tour_planner_env
```
