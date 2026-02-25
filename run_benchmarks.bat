@echo off
REM ============================================================
REM Module Benchmark Suite - Master Runner
REM ============================================================
REM Runs all module benchmarks sequentially and reports results.
REM 
REM Usage: run_benchmarks.bat
REM ============================================================

echo.
echo ============================================================
echo           MODULE BENCHMARK SUITE
echo ============================================================
echo.

set TOTAL_PASS=0
set TOTAL_FAIL=0

REM M1: Prompt Parser
echo [1/4] Running M1 Prompt Parser Benchmark...
echo ------------------------------------------------------------
python tests/benchmark_m1.py
if %ERRORLEVEL% EQU 0 (
    echo [M1] PASSED
    set /a TOTAL_PASS+=1
) else (
    echo [M1] FAILED
    set /a TOTAL_FAIL+=1
)
echo.

REM M2: Scene Planner
echo [2/4] Running M2 Scene Planner Benchmark...
echo ------------------------------------------------------------
python tests/benchmark_m2.py
if %ERRORLEVEL% EQU 0 (
    echo [M2] PASSED
    set /a TOTAL_PASS+=1
) else (
    echo [M2] FAILED
    set /a TOTAL_FAIL+=1
)
echo.

REM M3: Asset Generator
echo [3/4] Running M3 Asset Generator Benchmark...
echo ------------------------------------------------------------
python tests/benchmark_m3.py
if %ERRORLEVEL% EQU 0 (
    echo [M3] PASSED
    set /a TOTAL_PASS+=1
) else (
    echo [M3] FAILED
    set /a TOTAL_FAIL+=1
)
echo.

REM M4: Motion Generator
echo [4/4] Running M4 Motion Generator Benchmark...
echo ------------------------------------------------------------
python tests/benchmark_m4.py
if %ERRORLEVEL% EQU 0 (
    echo [M4] PASSED
    set /a TOTAL_PASS+=1
) else (
    echo [M4] FAILED
    set /a TOTAL_FAIL+=1
)
echo.

REM Summary
echo ============================================================
echo                    BENCHMARK SUMMARY
echo ============================================================
echo   Modules Passed: %TOTAL_PASS%/4
echo   Modules Failed: %TOTAL_FAIL%/4
echo ============================================================
echo.

if %TOTAL_FAIL% EQU 0 (
    echo ALL BENCHMARKS PASSED!
    exit /b 0
) else (
    echo SOME BENCHMARKS FAILED - Review above for details.
    exit /b 1
)
