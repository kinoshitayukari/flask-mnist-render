import numpy as np
from flask import Flask, render_template, request

from tensorflow.keras.models import load_model
from PIL import Image

SIZE = 54

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
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.reshape(1, SIZE, SIZE, 1)
        pred = model.predict(arr)
        text = '{}が予測されました'.format(np.argmax(pred))
    return render_template('index.html', text=text)

