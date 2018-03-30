import sys
sys.path.insert(0, '../library/')
from flask import Flask, redirect, url_for, request, jsonify
from werkzeug.datastructures import ImmutableMultiDict
from werkzeug.utils import secure_filename
import os
import json
from PIL import Image, ImageChops, ImageOps
import datetime
import numpy
from neural_network import NeuralNetwork
import time

UPLOAD_FOLDER = './upload_images'
ALLOWED_EXTENSIONS = set(['PNG'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def scaleData(dataArray):
    ''' Scale data respective to max and min values '''
    min = numpy.amin(dataArray)
    max = numpy.amax(dataArray)
    for i in range(len(dataArray)):
        dataArray[i] = (dataArray[i] - min) / (max - min)
    return dataArray


def mapResult(output):
    file = open("../tests/data/emnist/emnist-digits-mapping.txt",
                "r").read().splitlines()
    map = []
    list = [name.split() for name in file if name]
    for entry in list:
        map.append(entry[1])
    return chr(int(map[output]))


def getResult(imageArray):
    nn = NeuralNetwork([784, 600, 10])
    nn.loadResult()
    nn.printSums()
    outputs = nn.query(scaleData(imageArray))
    return mapResult(numpy.argmax(outputs))


@app.route("/", methods=['POST'])
def queryNetwork():
    '''
    Expects black on white or black on transparent image. Image gets croped, rescaled, inverted and converted to array
    Then the array is put through the neural net and the prediction is returned
    '''
    file = request.files["image"]
    filename = secure_filename(file.filename)
    file.save(os.path.join(
        app.config['UPLOAD_FOLDER'], filename))

    image = ImageForNetwork(os.path.join(
        app.config['UPLOAD_FOLDER'], filename))
    image.process()
    print(getResult(image.getArray()))
    return getResult(image.getArray())


class ImageForNetwork:

    def __init__(self, path):
        self.image = Image.open(path)

    def process(self):
        return self.__removeTransparency().__applyColorFilter().__cropImageWhitespace().__resizeImage()

    def getArray(self):
        return numpy.asfarray(self.image).flatten()

    def __removeTransparency(self):
        bg = Image.new("RGB", self.image.size, (255, 255, 255))
        bg.paste(self.image, (0, 0), self.image)
        self.image = bg
        return self

    def __applyColorFilter(self):
        # Invert, because in the MINST dataset 0 is white and 255 is black while here 0 is black
        self.image = ImageOps.invert(self.image)
        # Grayscale
        self.image = self.image.convert('L')
        return self

    def __cropImageWhitespace(self):

        # CROP WHITESPACE
        bg = Image.new(self.image.mode, self.image.size,
                       self.image.getpixel((0, 0)))
        diff = ImageChops.difference(self.image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            self.image = self.image.crop(bbox)
        return self

    def __resizeImage(self):

        # RESIZE TO MAX 20
        self.image.thumbnail((20, 20), Image.ANTIALIAS)

        # CALCULATE OFFSET BASED ON WIDTH AND HEIGHT
        temp_image = Image.new("L", (28, 28))
        widthOffset = 0
        if self.image.size[0] < 20:
            widthOffset = int(round((20 - self.image.size[0])/2))
        heightOffset = 0
        if self.image.size[1] < 20:
            heightOffset = int(round((20 - self.image.size[1])/2))

        # CALCULATE CENTER OF MASS
        xWeightDistanceSum = 0
        xWeightSum = 0
        yWeightDistanceSum = 0
        yWeightSum = 0
        arr = numpy.asfarray(self.image)
        for j, row in enumerate(arr):
            for i, col in enumerate(row):
                xWeightDistanceSum += i * arr[j][i]
                xWeightSum += arr[j][i]
                yWeightDistanceSum += j * arr[j][i]
                yWeightSum += arr[j][i]

        # APPLY CENTER OF MASS TO OFFSET (Not working so great)
        # widthOffset += int(round(self.image.size[0]/2 -
                #   (xWeightDistanceSum/xWeightSum)))
       # heightOffset += int(round(self.image.size[1]/2 -
                # (yWeightDistanceSum/yWeightSum)))

        # PLACE CROPPED 20x20 IMAGE IN 28x28
        temp_image.paste(self.image, ((4 + widthOffset, 4 + heightOffset)))
        self.image = temp_image
        self.image.save("./processing_images/guess" +
                        str(time.time()) + ".png")
        return self


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
