@echo off
:menu
cls
echo SimpleLM the Local LLM powered by Llamacpp
echo ===================
echo 1. Launch with Vulkan
echo 2. Launch with CPU
echo 3. Exit
echo ===================

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo Launching Llamacpp with Vulkan
    python launch_vulkan.py
    echo Launch succeed
    pause
    exit
)

if "%choice%"=="2" (
    echo Starting Python script with CPU...
    python launch_cpu.py
    echo Launch succeed
    pause
    exit
)

if "%choice%"=="3" (
    echo Exiting program...
    exit
) else (
    echo Invalid choice. Please try again.
    pause
    goto menu
)