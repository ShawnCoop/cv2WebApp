from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

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


def process_frames():
    global capture
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        while True:
            success, image = camera.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            #Convert image to RGB format
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            #Process the fram with MediaPipe Hand Landmarker
            results = hands.process(image_rgb)
            
            #Draw the hand landmarks on the image
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                for hand_landmarks in results.multi_hand_landmarks.landmark:
                    x_max = 0
                    y_max = 0
                    x_min = 0
                    y_min = 0
                    print(landmarks)
                    """
                    EHH not really working yet
                    for lm in handLMs.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
                    """
                    
            if capture:
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(['img_captures', "capture{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, image)
            
            ret, buffer = cv2.imencode('.jpg', cv2.flip(image, 1))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

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
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    