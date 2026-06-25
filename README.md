# CORTEX — Kuka Cable Simulation Environment

## Quick Start (No Installation Required)

A pre-built virtual environment (`venv/`) is included with all dependencies already installed — including **PyBullet** (which normally requires a C compiler to build from source).

### 1. Activate the environment

Open a terminal in this folder and run:

```powershell
.\venv\Scripts\Activate.ps1
```

> **CMD users:** Use `.\venv\Scripts\activate.bat` instead.

### 2. Run the code

```powershell
python kuka_env.py
```

### 3. Deactivate when done

```powershell
deactivate
```

---

## Important Notes

- **Python version:** This environment was built with **Python 3.11.8**. Your colleague must have the same **Python 3.11.x** installed on their machine for the venv to work.
- **OS:** This venv is Windows-only. macOS/Linux users will need to recreate it.
- If Python 3.11 is not available, install it from [python.org](https://www.python.org/downloads/) and then recreate the environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dependencies

| Package      | Version | Purpose                          |
|-------------|---------|----------------------------------|
| pybullet    | 3.2.7   | Physics simulation engine        |
| numpy       | 2.4.4   | Numerical computing              |
| matplotlib  | 3.10.7  | Plotting and visualization       |
