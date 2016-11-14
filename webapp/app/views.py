from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from app import app
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map, icons
import pickle
from sklearn.neighbors import KNeighborsClassifier as knn
from skimage.io import imread
import numpy as np
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(current_dir, 'model.sav'), "rb"))

markers = [
    {
        'icon': '//maps.google.com/mapfiles/ms/icons/green-dot.png',
        'lat': 23.195746,
        'lng': 72.632920,
        'infobox': "Hello I am <b style='color:green;'>GREEN</b>!"
    },
    {
        'icon': '//maps.google.com/mapfiles/ms/icons/blue-dot.png',
        'lat': 23.186004,
        'lng': 72.631472,
        'infobox': "Hello I am <b style='color:blue;'>BLUE</b>!"
    },
    {
        'icon': icons.dots.yellow,
        'title': 'Click Here',
        'lat': 23.183353,
        'lng': 72.629173,
        'infobox': (
            "Hello I am <b style='color:#ffcc00;'>YELLOW</b>!"
            "<h2>It is HTML title</h2>"
            "<img src='//placehold.it/50'>"
            "<br>Images allowed!"
        )
    }
]

def read_images(path, numImages, imageSize):
    x = np.zeros((numImages, imageSize))

    for idImage in range(numImages):
        fileName = "{0}/uploads/Image{1}.Bmp".format(path, idImage)
        image = imread(fileName, as_grey=True)

        x[idImage, :] = np.reshape(image, (1, imageSize))

    return x

def recognize(path=current_dir, numImages=2, imageSize=400):
    x = read_images(path, numImages, imageSize)
    val = ''.join(map(chr, model.predict(x)))
    print("Answer:", val)
    return val

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def clearUploadFolder():
    folder_path = app.config['UPLOAD_FOLDER']
    for file_object in os.listdir(folder_path):
        file_object_path = os.path.join(folder_path, file_object)
        if os.path.isfile(file_object_path):
            os.unlink(file_object_path)
        else:
            shutil.rmtree(file_object_path)

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    latitude = 23.1890566
    longitude = 72.6220352

    if request.method == 'POST':
        latitude = float(request.form['inputlat'])
        longitude = float(request.form['inputlong'])
        num_of_images = int(request.form['num-of-images'])
        files = []
        for i in range(num_of_images):
            files.append(request.files['image-file' + str(i)])

        clearUploadFolder()

        for i, _file in enumerate(files):
            if _file and allowed_file(_file.filename):
                filename = secure_filename(_file.filename)
                _file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                imageFormat = _file.filename.rsplit('.', 1)[1]
                os.rename(os.path.join(app.config['UPLOAD_FOLDER'], filename), os.path.join(app.config['UPLOAD_FOLDER'], 'Image' + str(i) + '.' + imageFormat))

        newMarker = {
            'icon': icons.dots.yellow,
            'title': recognize(numImages=num_of_images),
            'lat': latitude,
            'lng': longitude,
        }
        markers.append(newMarker)

    mymap = Map(
        identifier="mymap",
        varname="mymap",
        style=(
            "height:100%;"
            "width:100%;"
            "top:0;"
            "left:0;"
            "position:absolute;"
            "z-index:-1;"
        ),
        lat=latitude,
        lng=longitude,
        markers=markers
    )
    return render_template('index.html', mymap=mymap)

@app.route('/upload', methods=['POST'])
def upload():
    latitude = request.form['inputlat']
    longitude = request.form['inputlong']
    num_of_images = int(request.form['num-of-images'])
    files = []
    for i in range(num_of_images):
        files.append(request.files['image-file' + str(i)])

    clearUploadFolder()

    for i, _file in enumerate(files):
        if _file and allowed_file(_file.filename):
            filename = secure_filename(_file.filename)
            _file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            imageFormat = _file.filename.rsplit('.', 1)[1]
            os.rename(os.path.join(app.config['UPLOAD_FOLDER'], filename), os.path.join(app.config['UPLOAD_FOLDER'], 'Image' + str(i) + '.' + imageFormat))

    return redirect(url_for('index'))
