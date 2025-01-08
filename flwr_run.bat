@echo off
setlocal enabledelayedexpansion

:: Setting environment variables for lists
set "clients_list=10 5 20"
set "value_weight_list=0 1"

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

set RAY_DEDUP_LOGS=0

:: Loop through environments and their rounds
for %%e in (PendulumFixPos-v0 MountainCarFixPos-v0 CartPoleSwingUpFixInitState-v1) do (
    set "rounds=0"
    if "%%e"=="PendulumFixPos-v0" set "rounds=20"
    if "%%e"=="MountainCarFixPos-v0" set "rounds=80"
    if "%%e"=="CartPoleSwingUpFixInitState-v1" set "rounds=80"
    
    :: Loop through clients
    for %%c in (%clients_list%) do (
        :: Loop through value weights
        for %%v in (%value_weight_list%) do (
            :: Generate save directory path
            set "save_dir=multiagent/!time_str!_c%%c_%%e_VW%%v"
            
            echo Running experiment with:
            echo Environment: %%e
            echo Rounds: !rounds!
            echo Clients: %%c
            echo Value Weight: %%v
            echo Save Directory: !save_dir!
            echo.
            
            python .\modify_num.py -c %%c
            flwr run . --run-config "environment='%%e' num-server-rounds=!rounds! clients=%%c value_weight=%%v save_dir='!save_dir!'" local-simulation-gpu
            
            :: Add a small delay between runs
            timeout /t 2 /nobreak > nul
            echo.
            echo ------------------------
            echo.
        )
    )
)

echo All experiments completed.
pause