from flask import render_template
from app import app
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map, icons

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
@app.route('/index')
def index():
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
        lat=23.187148,
        lng=72.629544,
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
        ],
        # maptype = "TERRAIN",
        # zoom="5"
    )
    return render_template('index.html', mymap=mymap)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('index'))
