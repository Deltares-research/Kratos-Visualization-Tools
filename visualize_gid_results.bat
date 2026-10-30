@echo off
setlocal
set "PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON=%~dp0.venv\Scripts\python.exe"

if "%~1"=="" goto prompt
"%PYTHON%" "%~dp0visualize_gid_results.py" %*
set "RESULT=%ERRORLEVEL%"
goto finish

:prompt
echo Enter or drag a GiD mesh or results file into this window, then press Enter.
set "INPUT_PATH="
set /p "INPUT_PATH=GiD file or case base path: "
if not defined INPUT_PATH goto missing
"%PYTHON%" "%~dp0visualize_gid_results.py" "%INPUT_PATH:"=%"
set "RESULT=%ERRORLEVEL%"
goto finish

:missing
echo ERROR: No GiD file or case base path was supplied.
set "RESULT=1"

:finish
if not "%RESULT%"=="0" pause
exit /b %RESULT%
