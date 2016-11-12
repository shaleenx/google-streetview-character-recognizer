from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from app import app
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map, icons

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/r')
def r():
    print(request)

@app.route('/')
@app.route('/index')
def index():
    markers = []
    latitude = 23.1890566
    longitude = 72.6220352

    if request.method == 'POST':
        latitude = float(request.form['inputlat'])
        longitude = float(request.form['inputlong'])
        num_of_images = int(request.form['num-of-images'])
        files = request.files['image-file']

        if files and allowed_file(files.filename):
            filename = secure_filename(files.filename)
            files.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return redirect(url_for('index'))

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
        markers=[
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
    )
    return render_template('index.html', mymap=mymap)

@app.route('/upload', methods=['POST'])
def upload():
    latitude = request.form['inputlat']
    longitude = request.form['inputlong']
    num_of_images = request.form['num-of-images']
    files = request.files['image-file']

    print(request.files['image-file'])

    if files and allowed_file(files.filename):
        filename = secure_filename(files.filename)
        files.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return redirect(url_for('index'))
