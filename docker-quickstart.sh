#!/bin/bash
# Quick Docker Compose startup script for Linux/Mac

echo "Building Docker image..."
docker-compose build

echo ""
echo "Starting container..."
docker-compose up -d

echo ""
echo "Container is running!"
echo "Access the app at: http://localhost:7860"
echo ""
echo "To stop the container, run: docker-compose down"
echo "To view logs, run: docker-compose logs -f"
