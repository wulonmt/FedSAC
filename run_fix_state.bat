@echo off
setlocal enabledelayedexpansion

REM Check if environment name is provided
if "%1"=="" (
    echo Please provide an environment name as the first argument.
    exit /b 1
)

REM Check if number of agents is provided
if "%2"=="" (
    echo Please provide the number of agents as the second argument.
    exit /b 1
)

if "%3"=="" (
    echo Please provide the total rounds for server epoch as the third argument.
    exit /b 1
)

set env=%1
set agents=%2
set rounds=%3

echo Starting server
start /B python server.py -r %rounds% -c %agents%

timeout 3 > nul

echo %env%

set /a last_agent=%agents% - 1
for /l %%i in (0,1,%last_agent%) do (
    echo Starting client %%i
    start /B python FERclient_FixState.py -i %%i -e %env%
)

:loop
tasklist /FI "IMAGENAME eq python.exe" |find "python.exe" >nul
if not errorlevel 1 goto loop

echo All processes completed.
pause