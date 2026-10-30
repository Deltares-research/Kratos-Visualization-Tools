@echo off
setlocal
set "PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON=%~dp0.venv\Scripts\python.exe"

if "%~1"=="" goto prompt
"%PYTHON%" "%~dp0plot_saved_results.py" %*
set "RESULT=%ERRORLEVEL%"
goto finish

:prompt
echo Enter or drag a plot configuration JSON file into this window, then press Enter.
set "INPUT_PATH="
set /p "INPUT_PATH=Plot configuration: "
if not defined INPUT_PATH goto missing
"%PYTHON%" "%~dp0plot_saved_results.py" "%INPUT_PATH:"=%"
set "RESULT=%ERRORLEVEL%"
goto finish

:missing
echo ERROR: No plot configuration was supplied.
set "RESULT=1"

:finish
if not "%RESULT%"=="0" pause
exit /b %RESULT%
