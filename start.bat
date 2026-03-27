@echo off
setlocal
echo =======================================================
echo     Starting StintEngine Backend Server (Docker)
echo =======================================================
echo.

WHERE docker >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not installed or not running.
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop/
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo Building and starting the container in the background...
docker-compose up --build -d

echo.
echo =======================================================
echo   StintEngine Backend is running!
echo   API URL: http://localhost:5000
echo =======================================================
echo.
echo   - To view live logs: docker-compose logs -f
echo   - To stop server:    docker-compose down
echo.
echo Press any key to exit this window (server will keep running)...
pause >nul
