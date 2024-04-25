from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import mediapipe as mp

global capture, rec_frame, rec, out, switch, camera
caputure = 0
rec = 0
switch = 0

app = Flask(__name__, template_folder='./templates')
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("error: Unable to open camera")

#Creating the image capture folder
try:
    os.mkdir('./img_captures')
except OSError as error:
    print('Error creating folder for images')
    files = os.listdir('./img_captures')
    for file in files:
        if file.startswith("capture"):
            file_path = os.path.join("./img_captures", file)
            os.remove(file_path)
    print("the img_captures folder has been cleared")
    pass


def capture_Frame(capture_flag):  # generate frame by frame from camera
    global out, capture
    while True:
        success, frame = camera.read() 
        if success:
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['img_captures', "capture{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            if(capture_flag):
                try:
                    ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    pass       
        else:
            pass

def toggle_Camera():
    global camera
    if request.method == 'POST':
        if request.form.get('toggle_Camera') == 'Stop/Start':
            if camera.isOpened():
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
            

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global capture
    capture = 1
    return Response(capture_Frame(capture), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests', methods=['GET', 'POST'])
def tasks():
    global capture
    if request.method == 'POST':
        if request.form.get('capture') == 'Capture':
            global capture
            capture = 1
        if request.form.get('toggle_Camera') == 'Stop/Start':
            toggle_Camera()
    
    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    