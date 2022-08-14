from flask import Flask, render_template, request, session
from PIL import Image
import os
from werkzeug.utils import secure_filename
from flasgger import Swagger
from predict_1_img import seg
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', '.bmp'}
 
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
swagger = Swagger(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
app.secret_key = 'You Will Never Guess'

 
def detect_object(uploaded_image_path):
    output_path = seg(uploaded_image_path)
    return output_path
 
 
@app.route('/')
def index():
    return render_template('index_upload_and_display_image.html')
 
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
 
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)

        return render_template('index_upload_and_display_image_page2.html', user_image = session['uploaded_img_file_path'])
 
@app.route('/show_image')
def displayImage():
    img_file_path = session.get('uploaded_img_file_path', None)
    return render_template('show_image.html', user_image = img_file_path)
 
@app.route('/detect_object')
def detectObject():
    uploaded_image_path = session.get('uploaded_img_file_path', None)
    output_image_path= detect_object(uploaded_image_path)
    print(output_image_path)
    return render_template('show_image.html', user_image = output_image_path)
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
 
if __name__=='__main__':
    app.run(host = '0.0.0.0')