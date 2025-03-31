#!/bin/bash
# If PORT is not set, default to 8000
PORT=${PORT:-8000}

# Print the port for debugging
echo "Starting server on port: $PORT"

# Run with uvicorn directly
exec uvicorn main:app --host 0.0.0.0 --port $PORT
