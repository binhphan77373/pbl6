from flask import Flask, request, jsonify
import torch
from torchvision.transforms import transforms
from torchvision.models import resnet18
import torch.nn as nn
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
from PIL import Image

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

app = Flask(__name__)

tokenizer = SentenceTransformer('sentence-transformers/use-cmlm-multilingual')
model = Multimodal(Vision_model,Language_model,num_classes=16)
model.load_state_dict(torch.load('C:/Users/ADMIN/Documents/Deploy_PBL6/model_checkpoint/checkpoint/version_1/checkpoint.pth'))
model.eval()

class_labels = {0: 'Cell Phone', 1: 'Chair', 2: 'Digital + Camera', 3: 'Fridge', 4: 'Headphone', 
                5: 'Iron', 6: 'Keyboard', 7: 'Lamp', 8: 'Laptop', 9: 'Mouse', 
                10: 'Printer', 11: 'Speaker', 12: 'Table', 13: 'Tablet', 
                14: 'Television', 15: 'Vaccuum'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text_input = request.form['text_input']
        image_file = request.files['image_input']

        text_encoded = tokenizer.encode(text_input, convert_to_tensor=True).unsqueeze(0) 

        image = Image.open(image_file).convert('RGB')  
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image_tensor = image_transform(image).unsqueeze(0)  

        with torch.no_grad():
            prediction = model(image_tensor, text_encoded)

        prediction = prediction.argmax().item()
        predicted_label = class_labels.get(prediction, 'Unknown Class')

        return render_template('predict.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8080)