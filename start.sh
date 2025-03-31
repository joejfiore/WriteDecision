#!/bin/bash
# Print the port for debugging
echo "PORT: $PORT"

# Run with uvicorn directly
exec uvicorn main:app --host 0.0.0.0 --port $PORT