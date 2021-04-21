import os

from flask import Flask, render_template, request, Response
from tensorflow import keras

from sort import sort
from class_names import CLASS_NAMES_W

app = Flask(__name__)
model = keras.models.load_model('myModel/inceptionv3-transfer-94')
class_names = CLASS_NAMES_W


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/upload_sort', methods=['GET'])
def upload_sort():
    return render_template("picupload.html")


@app.route('/img/<filename>', methods=['GET'])
def img(filename):
    image = open("static/upload/img/{}".format(filename))
    resp = Response(image, mimetype="image/jpeg")
    return resp


# 表单提交路径，需要指定接受方式
@app.route('/getImg/', methods=['GET', 'POST'])
def getImg():
    imgData = request.files["image"]
    path = "static/upload/img/"
    imgName = imgData.filename
    file_path = path + imgName
    imgData.save(file_path)
    url = 'static/upload/img/' + imgName
    ls = sort(url, model)
    data = dict()
    for i in range(12):
        data.update({class_names[i]: float('%.2f' % (ls[i]*100))})
    sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
    return render_template("result.html", data=sorted_data, pic_name=imgName)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
