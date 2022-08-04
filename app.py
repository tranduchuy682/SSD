from flask import Flask, render_template, request, session
from PIL import Image
import os
from werkzeug.utils import secure_filename

from detect import detect
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', '.bmp'}
 
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
app.secret_key = 'You Will Never Guess'
 
def detect_object(uploaded_image_path):
 
    output_path={}
    runtimes={}
    for bb in ['mobilenetv3','resnet18','vgg16']:
        img_path = uploaded_image_path    
        
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB')
        output_path[bb] = os.path.join(app.config['UPLOAD_FOLDER'], "annotated_image_"+bb+".jpg")
        img, runtime = detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200, bb = bb)
        img.save(output_path[bb])
        runtimes[bb]=runtime
    return(output_path, runtimes)
 
 
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
    output_image_path, runtimes = detect_object(uploaded_image_path)
    print(output_image_path)
    return render_template('show_image.html', user_image_mobilenetv3 = output_image_path['mobilenetv3'], runtime_m = runtimes['mobilenetv3'],
                                            user_image_resnet18 = output_image_path['resnet18'],runtime_r = runtimes['resnet18'],
                                            user_image_vgg16 = output_image_path['vgg16'],runtime_v = runtimes['vgg16'])
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
 
if __name__=='__main__':
    app.run(debug = True)