@echo off
rem Starts the ErgoTwin camera module, then open http://localhost:8000
rem Extra arguments are passed on, for example:  run.cmd --source videos\demo.mp4
cd /d "%~dp0"
".venv\Scripts\python.exe" -m ergotwin %*
