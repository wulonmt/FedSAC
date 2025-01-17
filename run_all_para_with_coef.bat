@echo off
setlocal enabledelayedexpansion

:: Setting environment variables for lists
set "clients_list=5 10 20"
:: set "clients_list=10"
set "value_weight_list=0 1"
:: set "value_weight_list=1"
:: set "environments=CartPoleSwingUpFixInitState-v1 PendulumFixPos-v0 MountainCarFixPos-v0 HopperFixLength-v0"
set "environments=HopperFixLength-v0"

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

:: Loop through clients
for %%c in (%clients_list%) do (
    
    :: Loop through environments and their rounds
    for %%e in (%environments%) do (
        set "rounds=10"
        if "%%e"=="PendulumFixPos-v0" set "rounds=50"
        if "%%e"=="MountainCarFixPos-v0" set "rounds=100"
        if "%%e"=="CartPoleSwingUpFixInitState-v1" set "rounds=150"
        if "%%e"=="HopperFixLength-v0" set "rounds=600"
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
            
            :: Start server in a new PowerShell window
            echo Starting VW server
            start /B python VWserver.py -r !rounds! -c %%c --log_dir !save_dir!

            echo Server started, waiting for 3 seconds...
            timeout 5 > nul

            echo Starting clients...
            set /a last_agent=%%c - 1
            
            :: Start all clients in separate PowerShell windows
            for /l %%i in (0,1,!last_agent!) do (
                echo Starting client %%i
                start /B python FERclient_FixState.py -i %%i -e %%e --value_weight %%v --log_dir !save_dir!
            )

            :: Start checking script and wait for it to complete
            echo Waiting for server to complete...
            start /wait powershell -NoExit -Command "conda activate FedVW ; python check_server_end.py --log_dir !save_dir! ; exit"
            
            echo Current experiment completed.
            echo.
            echo ------------------------
            echo.
            
            :: Add a delay between experiments
            timeout /t 5 /nobreak > nul
            
        )
    )
)

:: Clean up any remaining marker file
if exist experiment_running.txt del experiment_running.txt

echo All experiments completed.
pause