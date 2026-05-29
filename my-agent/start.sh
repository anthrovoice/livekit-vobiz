#!/bin/bash
set -e

# Start the agent in the background
uv run python src/agent.py start &

# Start the trigger HTTP server in the foreground
uv run python src/trigger.py
