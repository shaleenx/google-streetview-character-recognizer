# google-streetview-character-recognizer

Dataset is stored in data/
Model Training Scripts are stored in models/
Code for Webapp is contained in webapp/
The Benchmark accuracy and training times are written in benchmark_vals.txt

To run the app, run train-model.py. It will store the model in webapp/app/model.sav.
Now navigate to execute
'''
python webapp/run.py
'''
to run the app. The app will use webapp/app/recognize.py which will use webapp/app/model.sav to predict input images.
