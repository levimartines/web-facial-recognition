import time
import cv2
from flask import Flask, render_template, Response

detector_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer.read("fisher_classifier.yml")
width, height = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    url = "http://192.168.25.60:4747/video"
    cap = cv2.VideoCapture(url)

    while cap.isOpened():
        ret, img = cap.read()
        image_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = detector_face.detectMultiScale(image_grey, scaleFactor=1.5, minSize=(30, 30))
        if ret:
            for (x, y, l, a) in detected_faces:
                image_face = cv2.resize(image_grey[y:y + a, x:x + l], (width, height))
                cv2.rectangle(img, (x, y), (x + l, y + a), (0, 0, 255), 2)
                class_id, confidence = recognizer.predict(image_face)
                if class_id == 1:
                    nome = "Levi - ID: " + str(class_id)
                elif class_id == 2:
                    nome = "Renan - ID: " + str(class_id)
                elif class_id == 3:
                    nome = "Sandra - ID: " + str(class_id)
                else:
                    nome = "Unknow - ID: " + str(class_id)
                cv2.putText(img, nome, (x, y + (a + 30)), font, 2, (0, 0, 255))
                cv2.putText(img, str(confidence), (x, y + (a + 50)), font, 1, (0, 0, 255))

            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.1)
        else:
            break


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
