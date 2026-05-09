@echo off
setlocal
pushd "%~dp0"
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 "%~dp0gui.py"
) else (
    python "%~dp0gui.py"
)
popd
endlocal

