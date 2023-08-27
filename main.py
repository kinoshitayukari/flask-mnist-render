import numpy as np
from flask import Flask, render_template, request

from tensorflow.keras.models import load_model
from PIL import Image

SIZE = 28

app = Flask(__name__)

model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/send', methods=['POST'])
def predict():
    image = request.files.get("file")
    text = 'ファイルが正しく選択されていません'
    if image:
        img = Image.open(image).convert('L').resize((SIZE, SIZE))
        pred = model.predict(np.array(img).reshape(1, SIZE, SIZE))
        text = '{}が予測されました'.format(np.argmax(pred))
    return render_template('index.html', text=text)

