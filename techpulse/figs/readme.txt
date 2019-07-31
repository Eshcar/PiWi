# for running draw_charts.py first install python 3
brew install python3
# then create virtual environment to install all libraries 
cd figs
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# finaly run the script
python ./draw_charts.py
