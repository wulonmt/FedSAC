@echo off
setlocal enabledelayedexpansion

:: 先執行 Python 腳本生成變數
python .\utils\path_extractor.py

:: 讀取生成的比較組合
set /p compare_dirs=<compare_pairs.txt

:: 處理每一組目錄
for %%a in ("%compare_dirs:;=";"%") do (
    for /f "tokens=1-5" %%1 in (%%a) do (
        echo Processing directories:
        echo %%1
        echo %%2
        echo -----------------------
        echo Generating results:
        echo %%3
        echo %%4
        echo %%5
        echo.
        
        :: 執行三個命令
        python .\utils\smooth_chart_multi_dir.py -l "%%1" "%%2" -n value_weighted uniform -s results/%%3
        python .\utils\smooth_chart_one_server.py -l "%%1" -s results/%%4
        python .\utils\smooth_chart_one_server.py -l "%%2" -s results/%%5
        
        echo.
        echo ------------------------
        echo.
        
        :: 添加短暫延遲
        timeout /t 2 /nobreak > nul
    )
)

echo All charts have been generated.
pause