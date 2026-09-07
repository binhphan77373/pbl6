from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import torch
from torchvision.models import resnet18
import torch.nn as nn
import os
from dataclasses import dataclass
from PIL import Image
import io
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = resnet18(pretrained=True)
model.fc

class Vision_model(nn.Module):
    def __init__(self, output_features=128):
        super(Vision_model, self).__init__()

        self.resnet18 = resnet18(pretrained=False)

        modelOutputFeats = self.resnet18.fc.in_features

        self.resnet18.fc = nn.Linear(modelOutputFeats,output_features)
    def forward(self, x):

        x = self.resnet18(x)
        return x

class Language_model(nn.Module):
    def __init__(self, input_size=768, output_features=128):
        super(Language_model, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 256,bias=True),
            nn.ReLU(),
            nn.Linear(256, output_features,bias=True),
        )

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.model(x)
        return x

class Multimodal(nn.Module):
    def __init__(self, Vision_model,Language_model,num_classes,output_features=128):
        super(Multimodal , self).__init__()
        self.vision_model = Vision_model()
        self.language_model = Language_model()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)

        self.fc1 = nn.Linear(output_features*2, 128,bias=True)
        self.fc2 = nn.Linear(128, num_classes,bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, X1, X2):
        X1 = self.vision_model(X1)
        X2 = self.language_model(X2)

        x = torch.cat([X1,X2],dim=1)
        transformer_encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=1)
        x = transformer_encoder(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

multimodal = Multimodal(Vision_model,Language_model,num_classes=16)

@dataclass(frozen=True)
class TrainingConfig:
  CHECKPOINT_DIR: str = os.path.join('model_checkpoint', 'checkpoint')
  
ckpt_version_dir = os.path.join(TrainingConfig.CHECKPOINT_DIR, 'version_1')
checkpoint_dir = TrainingConfig.CHECKPOINT_DIR
checkpoint_ckpt = os.path.join(ckpt_version_dir, 'checkpoint.pth')

# Load the entire model checkpoint
trained_generator = torch.load(checkpoint_ckpt, map_location=torch.device('cpu'))
model_state_dict = {}
for key, value in trained_generator.items():
    key = key.replace('vision_model.', '').replace('language_model.', '')
    if 'vision_model' in key:
        new_key = key.replace('vision_model.', '')
        if new_key in multimodal.vision_model.state_dict():
            model_state_dict['vision_model.' + new_key] = value
    elif 'language_model' in key:
        new_key = key.replace('language_model.', '')
        if new_key in multimodal.language_model.state_dict():
            model_state_dict['language_model.' + new_key] = value
    else:
        if key in multimodal.state_dict():
            model_state_dict[key] = value

multimodal.load_state_dict(trained_generator, strict=False)

# Load the text tokenizer
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)

# Map your output classes to labels
class_labels = {0: 'Cell Phone', 1: 'Chair', 2: 'Digital + Camera', 3: 'Fridge', 4: 'Headphone', 
                5: 'Iron', 6: 'Keyboard', 7: 'Lamp', 8: 'Laptop', 9: 'Mouse', 
                10: 'Printer', 11: 'Speaker', 12: 'Table', 13: 'Tablet', 
                14: 'Television', 15: 'Vaccuum'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_input = request.form['text_input']
    image_file = request.files['image_input']

    text_sequences = tokenizer.texts_to_sequences([text_input])
    text_padded = pad_sequences(text_sequences, maxlen=768)

    image = preprocess_image(image_file)
    text_padded = torch.tensor(text_padded, dtype=torch.long)
    print("Image dtype:", image.dtype)
    print("Text padded dtype:", text_padded.dtype)

    with torch.no_grad():
        predictions = multimodal(image,text_padded)

    predicted_class = torch.argmax(predictions).item()
    predicted_label = class_labels.get(predicted_class, 'Unknown Class')

    return render_template('predict.html', prediction=predicted_label)

def preprocess_image(image_file):
    img = image_file.read()
    img_array = img_to_array(Image.open(io.BytesIO(img)).convert('RGB').resize((224, 224)))
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.ascontiguousarray(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    img_array_copy = img_array.copy()
    return torch.tensor(img_array_copy, dtype=torch.float32)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8080)