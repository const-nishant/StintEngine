#!/bin/bash
echo "======================================================="
echo "    Starting StintEngine Backend Server (Docker)       "
echo "======================================================="
echo ""

if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed or not in PATH."
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

echo "Building and starting the container in the background..."
docker-compose up --build -d

echo ""
echo "======================================================="
echo "  StintEngine Backend is running!"
echo "  API URL: http://localhost:5000"
echo "======================================================="
echo ""
echo "  - To view live logs: docker-compose logs -f"
echo "  - To stop server:    docker-compose down"
echo ""
