from flask import Flask
from flask_googlemaps import GoogleMaps

app = Flask(__name__)
app.config.from_object('config')

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

from app import views

GoogleMaps(app, key='AIzaSyACFwNwWltDdXCzkZRMWu8cS9oy_QloB-Q')
