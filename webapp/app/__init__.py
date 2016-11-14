from flask import Flask
from flask_googlemaps import GoogleMaps
import os

app = Flask(__name__)
app.config.from_object('config')


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'Bmp'])

from app import views

GoogleMaps(app, key='AIzaSyACFwNwWltDdXCzkZRMWu8cS9oy_QloB-Q')
