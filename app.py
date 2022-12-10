from flask import Flask, render_template, Response
import cv2
import torch
import easyocr
from PIL import ImageFont, ImageDraw, Image, ImageGrab
import numpy as np


app = Flask(__name__)
camera = cv2.VideoCapture(0)


def plate_detection():
    path = 'best.pt'
    model = torch.hub.load('ultralytics/yolov5',
                           'custom', path, force_reload=True)

    capture = cv2.VideoCapture(0)

    while True:
        success, frame = capture.read()

        if not success:
            break
        else:
            results = model(frame)
            reader = easyocr.Reader(['bn'], gpu=True)
            result = reader.readtext(frame)
            frame = np.squeeze(results.render())

            text = ''
            for string in result:
                text = text+string[1]+'\n'

            fontpath = "banglamn.ttc"  # <== 这里是宋体路径
            font = ImageFont.truetype(fontpath, 24)
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            b, g, r, a = 221, 82, 6, 0
            draw.text((50, 80),  text, font=font, fill=(b, g, r, a))
            frame = np.array(img_pil)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(plate_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
