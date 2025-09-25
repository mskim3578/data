
#pip install opencv-python
from flask import Flask, render_template, Response, request
import pathlib
# import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL  import Image


app = Flask(__name__)
name="mnist 이미지 확인"

static = 'static/'

nummodel = load_model("mnist.h5")  #1
fasmodel = load_model("fashion.h5")  #2

class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
     'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  #3


def prediction(filename) :
    
    oneimg=Image.open(filename)
    oneimg=np.array(oneimg)
    oneimg=oneimg.reshape(1,28*28)/255 
   ####  4 
    if filename.find("fas") > 0 :
        results=fasmodel.predict(oneimg)
        pred_y = class_names[np.argmax(results[0])]
    else:
        results=nummodel.predict(oneimg)
        pred_y = np.argmax(results[0])
    ####   
    
    print(pred_y)
    return pred_y



@app.route('/', methods=['GET', 'POST'])
def index():
    global req_name
    current_dir = pathlib.Path().absolute()
    print(current_dir)
    if request.method != 'POST':
        return render_template('index.html', name=name) 
    else:
        
        file = request.files['upload']
        filename=file.filename
        file.save(static+filename)
        pred_y=prediction(static+filename)
        return render_template('index.html', image_file=filename, num=pred_y)



if __name__ == '__main__':
    app.run(debug=True, port="5002")
