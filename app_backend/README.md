# Metta App Backend

This is the backend for the app deployed at https://observatory.softmax-research.net/

## Requirements

- PostgreSQL 16 or newer (required for ANY_VALUE aggregate function)
- Python 3.11+

## Building the Docker Image

In the parent directory run the command

```
docker build -t metta-app-backend:latest -f app_backend/Dockerfile .
```

This will build the Docker image tagged as `metta-app-backend:latest`.
The command must be run from the parent directory because we are using the parent `uv.lock` file.

## Running the Container

```bash
docker run -p 8000:8000 \
  -e STATS_DB_URI="postgres://user:password@host:port/db" \
  metta-app-backend:latest
```

If you are running a postgres instance locally, use `host.docker.internal` as host

## Environment Variables

- `STATS_DB_URI`: PostgreSQL connection string (default: `postgres://postgres:password@127.0.0.1/postgres`)
- `HOST`: Server host (default: `127.0.0.1`)
- `PORT`: Server port (default: `8000`)
- `DEBUG_USER_EMAIL` : In production, your auth token is managed by oauth2_proxy. This is a way to run locally
without the oauth2_proxy. Set this to match the user_id of training runs you want to edit in the frontend
(e.g., `DEBUG_USER_EMAIL=test@example.com` allows editing runs created by that user).

## Development

To run locally without Docker:

1. Start PostgreSQL 16+ container:
```bash
docker run -d --name metta-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=postgres \
  -p 5432:5432 \
  postgres:16
```

2. Run the backend:
```bash
cd app_backend
uv run python server.py
```

## Testing with Sample Data

### Option 1: Synthetic Test Data

To create a test training run with synthetic data:

```bash
# Ensure backend is running on port 8001
cd app_backend
./create_test_training_run.py
```

This creates a complete training run with synthetic data including epochs, policies, and episodes.

### Option 2: Real Training System (Minimal)

To create a training run using the actual metta training system (recommended for testing integrations):

```bash
# Ensure backend is running on port 8001
# Get token and run minimal training in one go
METTA_API_KEY=$(curl -s -X POST http://localhost:8001/tokens \
  -H "Content-Type: application/json" \
  -H "X-Auth-Request-Email: test@example.com" \
  -d '{"name": "training_test_token"}' | jq -r '.token') \
./tools/train.py \
  run=test_git_hash \
  stats_server_uri=http://localhost:8001 \
  trainer.total_timesteps=2 \
  trainer.num_workers=1 \
  device=cpu \
  +hardware=macbook
```

This runs the real training pipeline with only 2 timesteps (takes ~30 seconds), creating an authentic training run that can be viewed in the observatory frontend. Useful for testing:
- Training run creation flow
- Integration between metta trainer and backend
- Git hash storage (when implemented)
- Frontend components with real data

**Note:** You may see WandB permission warnings - these are harmless and expected in local development.

## API Endpoints

- `/dashboard/*` - Dashboard-related endpoints
- `/stats/*` - Statistics and data recording endpoints
