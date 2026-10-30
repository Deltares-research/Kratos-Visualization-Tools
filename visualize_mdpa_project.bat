@echo off
setlocal
set "PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON=%~dp0.venv\Scripts\python.exe"

if "%~1"=="" goto prompt
"%PYTHON%" "%~dp0visualize_mdpa_project.py" %*
set "RESULT=%ERRORLEVEL%"
goto finish

:prompt
echo Enter or drag an MDPA file into this window, then press Enter.
set "INPUT_PATH="
set /p "INPUT_PATH=MDPA file: "
if not defined INPUT_PATH goto missing
"%PYTHON%" "%~dp0visualize_mdpa_project.py" "%INPUT_PATH:"=%"
set "RESULT=%ERRORLEVEL%"
goto finish

:missing
echo ERROR: No MDPA file was supplied.
set "RESULT=1"

:finish
if not "%RESULT%"=="0" pause
exit /b %RESULT%
