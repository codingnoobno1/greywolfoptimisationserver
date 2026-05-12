@echo off
echo Starting Greywolf AI Platform setup...

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating venv and installing dependencies...
call .\\venv\\Scripts\\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Starting server...
uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
