from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread

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


def  capture_Frame():
    global out, capture, rec_frame, camera
    
    while True:
        success, frame = camera.read()
        if success:
            if capture: #if capture is 1 -> clicked, take that frame with a new datetime and put it in img_captures
                capture = 0
                now = datetime.datetime.now()
                p =  os.path.sep.join(['img_captures', "capture_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            elif (request.form.get('stop') == 'Stop/Start'): #if stop/start button is clicked and switch == 1, update it to 0 and stop the camera; if switch == 0 change it to 1 and create the camera object
                if (switch == 1):
                    switch = 0
                    camera.release()
                    cv2.destroyAllWindows()
                else:
                    camera = cv2.VideoCapture(0)
                    switch = 1

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
    print("inside the video feed method")
    return Response(capture_Frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests', methods=['GET', 'POST'])
def tasks():
    if request.method == 'POST':
        if request.form.get('capture') == 'Capture':
            global capture
            capture = 1
            print("Requests, Pos")
        if request.form.get('toggle_Camera') == 'Stop/Start':
            toggle_Camera()
    
    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    