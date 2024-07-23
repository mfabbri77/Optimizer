# virtual environment setup 
/usr/bin/python3 -m venv venv
source venv/bin/activate

#update pip
/usr/bin/python3 -m pip install --upgrade pip

# istallazione delle librerie necessarie
pip install -r requirements.txt

# esecuzione
python3 gui.py
