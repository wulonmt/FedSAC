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

if "%4"=="" (
    echo Please provide the kl divergence coef value.
    exit /b 1
)

if "%5"=="" (
    echo Please provide the entropy divergence coef value.
    exit /b 1
)

if "%6"=="" (
    echo Please provide add kl divergence regulization or not. 1 for True.
    exit /b 1
)

if "%7"=="" (
    echo Please provide use value as weight or not. 1 for True.
    exit /b 1
)

if "%8"=="" (
    echo Please provide server port.
    exit /b 1
)

set env=%1
set agents=%2
set rounds=%3
set kl=%4
set ent=%5
set add_kl=%6
set vw=%7
set port=%8

:: Get current date and time with wmic command to ensure consistent format
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"

:: Extract date and time components
set "year=%dt:~0,4%"
set "month=%dt:~4,2%"
set "day=%dt:~6,2%"
set "hour=%dt:~8,2%"
set "minute=%dt:~10,2%"

:: Create time string in format YYYY_MM_DD_HH_MM
set "time_str=%year%_%month%_%day%_%hour%_%minute%"

set "save_dir=multiagent/!time_str!_c!agents!_!env!_VW!vw!"

echo Starting VW server
start /B python VWserver.py -r %rounds% -c %agents% -p %port%

timeout 3 > nul

echo %env%

set /a last_agent=%agents% - 1
for /l %%i in (0,1,%last_agent%) do (
    echo Starting client %%i
    start /B python FERclient_FixState.py -i %%i -e %env% --kl_coef %kl% --ent_coef %ent% --add_kl %add_kl% --value_weight %vw% --log_dir %save_dir% -p %port%
)

:loop
tasklist /FI "IMAGENAME eq python.exe" |find "python.exe" >nul
if not errorlevel 1 goto loop

echo All processes completed.
pause