import os
import uuid
import urllib
from flask import Flask, render_template, request
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import model_from_json

app = Flask(__name__)
j_file = open('model.json', 'r')
loaded_json_model = j_file.read()
j_file.close()
model = model_from_json(loaded_json_model)
model.load_weights('model.h5')

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])
classes = [
    'Actinic Keratoses',
    'Basal Cell Carcinoma',
    'Benign Keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic Nevi',
    'Vascular naevus'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXT

def predict(filename, model):
    img = Image.open(filename).convert('RGB').resize((224, 224))
    img_array = img_to_array(img).reshape(1, 224, 224, 3).astype('float32') / 255.0
    result = model.predict(img_array)

    dict_result = {classes[i]: result[0][i] for i in range(7)}

    sorted_results = sorted(dict_result.items(), key=lambda x: x[1], reverse=True)
    class_result = [item[0] for item in sorted_results[:3]]
    prob_result = [(item[1] * 100).round(2) for item in sorted_results[:3]]

    return class_result, prob_result


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/success', methods=['POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/images')
    if request.method == 'POST':
        if request.form:
            link = request.form.get('link')
            try:
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                with open(img_path, "wb") as output:
                    output.write(resource.read())
                img = filename

                class_result, prob_result = predict(img_path, model)

                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                }

            except Exception as e:
                print(str(e))
                error = 'Cant access this image'

            if not error:
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)

        elif request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                img = file.filename

                class_result, prob_result = predict(img_path, model)

                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                }

            else:
                error = "Bro? I can't access this image lmao"

            if not error:
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')