echo Creating venv...
python3 -m venv stock_predictor

echo Activating venv...
source stock_predictor/bin/activate

echo Installing requirements...
python3 -m pip install -r requirements.txt

# If the OS isn't Mac OS, install tensorflow-gpu
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo Installing tensorflow-gpu...
    
    python3 -m pip install tensorflow-gpu=2.2.0
fi

echo venv creation complete.