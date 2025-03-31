#!/bin/bash
# Make this file executable
# chmod +x start.sh

# Print the port for debugging
echo "PORT: $PORT"

# Run with uvicorn directly - this is the most reliable method for Render
exec uvicorn main:app --host 0.0.0.0 --port $PORT
