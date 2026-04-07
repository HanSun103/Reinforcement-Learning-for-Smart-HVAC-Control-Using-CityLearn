@echo off
REM ==========================================================================
REM install.bat — CityLearn HVAC RL Project (Windows, Anaconda)
REM ==========================================================================
REM
REM IMPORTANT: Run from an ANACONDA PROMPT (not regular PowerShell).
REM   Start → search "Anaconda Prompt" → open it
REM   Then: cd to the project folder and type:  install.bat
REM
REM This creates a FRESH conda environment called "citylearn-rl".
REM Using a dedicated env avoids DLL conflicts with the base environment.
REM ==========================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo  CityLearn HVAC RL — Setup (Anaconda Prompt required)
echo ============================================================
echo.

REM --------------------------------------------------------------------------
REM Step 1: Create a fresh conda environment
REM --------------------------------------------------------------------------
REM A dedicated env isolates all packages and avoids DLL/TBB/MKL conflicts
REM that affect pip-installed PyTorch in the base Anaconda environment.
REM --------------------------------------------------------------------------
echo [1/6] Creating conda environment "citylearn-rl" (Python 3.11)...
call conda create -n citylearn-rl python=3.11 -y
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Could not create conda environment. Is Anaconda Prompt open?
    pause & exit /b 1
)

REM --------------------------------------------------------------------------
REM Step 2: Activate the environment
REM --------------------------------------------------------------------------
echo.
echo [2/6] Activating conda environment "citylearn-rl"...
call conda activate citylearn-rl
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Could not activate the environment.
    pause & exit /b 1
)

REM --------------------------------------------------------------------------
REM Step 3: PyTorch (CPU-only) via conda
REM --------------------------------------------------------------------------
REM conda-installed PyTorch bundles the correct MKL/TBB/DLL dependencies.
REM pip-installed PyTorch frequently causes "DLL init failed" in conda envs.
REM --------------------------------------------------------------------------
echo.
echo [3/6] Installing PyTorch 2.3.1 (CPU-only) via conda...
echo       This may take 5-10 minutes. Do not close this window.
call conda install pytorch=2.3.1 cpuonly -c pytorch -y
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PyTorch install failed. Try running manually:
    echo   conda install pytorch=2.3.1 cpuonly -c pytorch -y
    pause & exit /b 1
)

REM --------------------------------------------------------------------------
REM Step 4: CityLearn (without openstudio)
REM --------------------------------------------------------------------------
REM CityLearn 2.5.0 declares openstudio as a dependency, but no PyPI wheel
REM exists for Python 3.11 on Windows. We install with --no-deps and add
REM only the packages we actually need.
REM --------------------------------------------------------------------------
echo.
echo [4/6] Installing CityLearn 2.5.0 (skipping openstudio)...
pip install citylearn==2.5.0 --no-deps
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CityLearn install failed.
    pause & exit /b 1
)

REM --------------------------------------------------------------------------
REM Step 5: CityLearn minimal deps + Stable Baselines3
REM --------------------------------------------------------------------------
REM gymnasium is pinned to 0.28.1 (CityLearn's maximum supported version).
REM SB3 is pinned to 2.2.1 (last version compatible with gymnasium 0.28).
REM SB3 is installed with --no-deps so it cannot auto-upgrade gymnasium.
REM platformdirs is a direct CityLearn runtime dep (missed by --no-deps).
REM --------------------------------------------------------------------------
echo.
echo [5/6] Installing gymnasium, SB3, and CityLearn dependencies...
pip install "gymnasium==0.28.1" "scikit-learn==1.2.2" simplejson pyyaml platformdirs
pip install "stable-baselines3==2.2.1" --no-deps
pip install cloudpickle
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Package install failed.
    pause & exit /b 1
)

REM --------------------------------------------------------------------------
REM Step 6: Utilities
REM --------------------------------------------------------------------------
REM numpy is capped at <2.0.0 — CityLearn 2.5.0 is not compatible with numpy 2.x.
REM matplotlib/pandas sometimes auto-install numpy 2.x; we pin it here after.
REM --------------------------------------------------------------------------
echo.
echo [6/6] Installing matplotlib, pandas, numpy...
pip install "matplotlib>=3.7.0" "pandas>=2.0.0"
pip install "numpy>=1.24.0,<2.0.0"

REM --------------------------------------------------------------------------
REM Verify all key packages load correctly
REM --------------------------------------------------------------------------
echo.
echo ============================================================
echo  Verifying installation...
echo ============================================================
python -c "import torch;            print('  torch    :', torch.__version__)"
python -c "import gymnasium;        print('  gymnasium:', gymnasium.__version__)"
python -c "import stable_baselines3;print('  sb3      :', stable_baselines3.__version__)"
python -c "import citylearn;        print('  citylearn:', citylearn.__version__)"
python -c "import simplejson;       print('  simplejson: OK')"
python -c "import sklearn;          print('  sklearn  :', sklearn.__version__)"

echo.
echo ============================================================
echo  Environment "citylearn-rl" is ready!
echo.
echo  NEXT STEPS:
echo.
echo  1. In Cursor/VS Code: press Ctrl+Shift+P, choose
echo       "Python: Select Interpreter"
echo     and pick:  citylearn-rl (conda)
echo.
echo  2. Run the quick smoke test:
echo       conda activate citylearn-rl
echo       python src/quick_test.py
echo.
echo  3. Train the model:
echo       python src/train.py
echo ============================================================
echo.
pause
