# -*- coding: utf-8 -*-


from flask import Flask, request, render_template
import pathlib
# templates

app = Flask(__name__)
name="mnist 이미지 확인"

static="static/" # upload folder



@app.route('/', methods=['GET', 'POST'])
def index():   
    current_dir = pathlib.Path().absolute()
    print(current_dir)
    if request.method != 'POST':
        return render_template('index.html', name=name) 
    else:
        file=request.files['upload']
        filename=file.filename
        print(filename)
        file.save(static+filename)
        return render_template('index.html', name=name, imgname=filename) 
    
    

if __name__ == '__main__':
    app.run(debug=True, port="5002")