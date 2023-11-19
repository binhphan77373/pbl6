from flask import Flask, render_template, request, jsonify
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the saved model
model_path = 'C:/Users/ADMIN/Documents/Deploy_PBL6/models/model_checkpoint.h5'
model = load_model(model_path)

# Load the text tokenizer
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)

# Map your output classes to labels
class_labels = {0: 'Class1', 1: 'Class2', 2: 'Class3', 3: 'Class4', 4: 'Class5', 5: 'Class6', 6: 'Class7', 7: 'Class8', 8: 'Class9', 9: 'Class10'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_input = request.form['text_input']
    image_file = request.form['image_input']

    text_sequences = tokenizer.texts_to_sequences([text_input])
    text_padded = pad_sequences(text_sequences, maxlen=100)

    image = preprocess_image(image_file)
    predictions = model.predict({'text_input': text_padded, 'image_input': image})
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels.get(predicted_class, 'Unknown Class')
    return predicted_label

    #return render_template('predict.html', prediction=predicted_label)

def preprocess_image(image_file):
    response = requests.get(image_file)
    img_array = img_to_array(Image.open(io.BytesIO(response.content)).convert('RGB').resize((224, 224)))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

if __name__ == '__main__':
    app.run(debug=True, host='localhost',port=8080)