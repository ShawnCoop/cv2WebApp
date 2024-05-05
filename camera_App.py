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

#Creating the image capture folder and subfolders for cropped : Just change the letter in the path to make a folder for whatever letter
try:
    os.mkdir('./img_captures//cropped//y')
    os.mkdir('./img_captures//uncropped//y')
except OSError as error:
    print('Error creating folder for images')
    files = os.listdir('./img_captures//cropped//y')
    for file in files:
        if file.startswith("y"):
            file_path = os.path.join('img_captures//cropped//y', file)
            os.remove(file_path)
    print("the img_captures folder has been cleared")
    files = os.listdir('./img_captures//uncropped//y')
    for file in files:
        if file.startswith("y"):
            file_path = os.path.join('img_captures//uncropped//y', file)
            os.remove(file_path)
    pass


def process_frames(): #Go down to  107 and 110 and change the name of the file from y to whatever letter
    global capture
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        while True:
            success, image = camera.read()
            image_no_draw = image.copy()
            if image is None:
                print("couldn't capture frame")
                continue
            crop_coordinates = []
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
                #Gets the min and max x and y coordinates for each image to crop: data set used cropped images
                for hand_landmarks in results.multi_hand_landmarks:
                    height, width, _ = image.shape
                    x_max, y_max, x_min, y_min = 0, 0, 10, 10
                    for point in hand_landmarks.landmark:
                        x = point.x
                        y = point.y
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                        
                        
                    pixel_x_max = int(x_max * width) + 30
                    pixel_y_max = int(y_max * height) + 30
                    pixel_x_min = int(x_min * width) - 30
                    pixel_y_min = int(y_min * height) - 30
                    crop_length = 0
                    if pixel_x_max - pixel_x_min > pixel_y_max - pixel_y_min:
                        crop_length = pixel_x_max - pixel_x_min
                    else:
                        crop_length = pixel_y_max - pixel_y_min
                    
                    crop_coordinates = [pixel_y_min, pixel_y_min + crop_length, pixel_x_min, pixel_x_min + crop_length]
                    
                    capture = 0
                    now = datetime.datetime.now()
                    pc = os.path.sep.join(['img_captures//cropped//y', "y_cropped{}.png".format(str(now).replace(":",''))])
                    cropped = image_no_draw[crop_coordinates[0]:crop_coordinates[1], crop_coordinates[2]:crop_coordinates[3]]
                    cv2.imwrite(pc, cropped)
                    p = os.path.sep.join(['img_captures//uncropped//y', "y_capture{}.png".format(str(now).replace(":",''))])
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
                #resize the frame with the new dimensions
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

def predict(filepath):
    image_path = filepath
    image = keras.utils.load_img(image_path, target_size = (64,64))
    input_data = keras.utils.img_to_array(image)
    input_data = np.expand_dims(input_data, axis=0)
    prediction = model.predict(input_data)

    classes = np.argmax(prediction, axis = 1)
    print(idx_to_letter(classes))

def idx_to_letter(idx):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    idx = idx[0]
    if 0 <= idx < len(letters):
        return letters[idx]
    if idx == 26:
        return 'nothing'
    if idx == 27:
        return '(space)'

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
    