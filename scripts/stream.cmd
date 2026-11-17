@echo off
rem Streams a video file in a loop as a live network feed, the way an IP camera would send it.
rem Usage:  scripts\stream.cmd videos\demo.mp4 [port]
rem Then start the source  udp://127.0.0.1:5000  in the page.
setlocal
if "%~1"=="" (
  echo Usage: scripts\stream.cmd ^<video file^> [port]
  exit /b 1
)
set "PORT=%~2"
if "%PORT%"=="" set "PORT=5000"

rem A terminal opened before ffmpeg was installed does not have it on its PATH yet,
rem so fall back to the folder winget installs it into.
set "FFMPEG=ffmpeg"
where ffmpeg >nul 2>nul
if errorlevel 1 (
  for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*") do (
    for /d %%B in ("%%D\ffmpeg-*") do if exist "%%B\bin\ffmpeg.exe" set "FFMPEG=%%B\bin\ffmpeg.exe"
  )
)

echo Streaming %~1 to udp://127.0.0.1:%PORT%  (Ctrl+C to stop)
"%FFMPEG%" -hide_banner -loglevel warning -re -stream_loop -1 -i "%~1" -an -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -pix_fmt yuv420p -f mpegts "udp://127.0.0.1:%PORT%?pkt_size=1316"
