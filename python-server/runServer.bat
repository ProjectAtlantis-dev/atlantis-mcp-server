@echo off
REM Auto-restart loop for server

:loop
echo Starting server...

python server.py ^
  --email=youremail@gmail.com ^
  --api-key=foobar ^
  --host=localhost ^
  --port=8000 ^
  --cloud-host=wss://projectatlantis.ai ^
  --cloud-port=443 ^
  --app-name=pa ^
  --service-name=home

set exit_code=%ERRORLEVEL%
echo Server exited with code %exit_code%

if %exit_code%==0 (
    echo Clean shutdown detected. Not restarting.
    goto end
)

echo Restarting in 2 seconds...
timeout /t 2 /nobreak >nul
goto loop

:end
echo Exiting restart loop.