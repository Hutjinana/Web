from flask import Flask, render_template, Response
import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

app = Flask(__name__)


def calculate_angle(a, b, c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle>180 :
        angle = 360-angle
    return angle

def generate_frames(model,display_text):
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    start_time = time.time()
    counter = 0
    prev_frame_time = 0
    new_frame_time = 0
    stage = None

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                row = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten().tolist()
                X = pd.DataFrame([row])
                physical_therapy_class = model.predict(X)[0]
                physical_therapy_prob = model.predict_proba(X)[0]

                max_prob = np.max(physical_therapy_prob)
                color = (0, 255, 0) if max_prob >= 0.70 and physical_therapy_class == "on" else (0, 0, 255)

                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

                if physical_therapy_class == "off":
                    stage = 'down'
                elif physical_therapy_class == "on" and stage == 'down':
                    stage = 'up'
                    counter += 1

                if physical_therapy_class == "on" and max_prob >= 0.70:
                    elapsed_time = int(time.time() - start_time)
                else:
                    start_time = time.time()
                    elapsed_time = counter

                new_frame_time = time.time()
                fps = int(1 / (new_frame_time - prev_frame_time))
                prev_frame_time = new_frame_time

                cv2.rectangle(image, (0, 0), (300, 60), (245, 117, 16), -1)
                cv2.putText(image, 'FPS', (270, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(fps), (260, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, physical_therapy_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(max_prob, 2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, display_text, (195, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(elapsed_time), (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

def generate_frames2():
    camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    prev_frame_time = 0
    new_frame_time = 0
    timer = 0
    fps = 0
    stage = None
    counter = 0

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                d = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                e = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                f = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                new_frame_time = time.time()
                fps = int(1 / (new_frame_time - prev_frame_time))
                prev_frame_time = new_frame_time

                angle_left = 360 - int(calculate_angle(d, e, f))
                color = (0, 0, 255)  # Red
                if 185 < angle_left < 210:
                    if stage == "down":
                        color = (0, 255, 0)  # Green
                        counter += 1
                        stage = "up"
                    else:
                        color = (0, 255, 0)  # Green
                else :
                    stage = "down" 

                    

                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                )

                cv2.rectangle(image, (0, 0), (300, 60), (245, 117, 16), -1)
                cv2.putText(image, 'FPS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(fps), (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, 'REPS', (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            except:
                pass

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/') #homepage
def home():
    return render_template('homepage.html')

@app.route('/Stomach')
def Stomach():
    return render_template('Stomach.html')

@app.route('/Stomach/start_video') #Stomach
def start_video_Stomach():
    global camera
    if not camera.isOpened():
        camera.open(0)
    return "Stomach Camera started"

@app.route('/Stomach/video_feed')
def video_feed_Stomach():
    with open('Sleep3.pkl', 'rb') as f:
        model = pickle.load(f)
    display_text = "TIME"
    return Response(generate_frames(model,display_text), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Stomach/stop_video')
def stop_video_Stomach():
    global camera
    if camera.isOpened():
        camera.release()
    return "Stomach Camera stopped"

@app.route('/Pillow') #Pillow
def Pillow():
    return render_template('Pillow.html')

@app.route('/Pillow/start_video')
def start_video_Pillow():
    global camera
    if not camera.isOpened():
        camera.open(0)
    return "Pillow Camera started"

@app.route('/Pillow/video_feed')
def video_feed_Pillow():
    with open('Sleep3.pkl', 'rb') as f:
        model = pickle.load(f)
    display_text = "TIME"
    return Response(generate_frames(model,display_text), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Pillow/stop_video')
def stop_video_Pillow():
    global camera
    if camera.isOpened():
        camera.release()
    return "Pillow Camera stopped"

@app.route('/Prone_elbow') #Prone on elbow
def Prone_elbow():
    return render_template('Prone_elbow.html')

@app.route('/Prone_elbow/start_video')
def start_video_Prone_elbow():
    global camera
    if not camera.isOpened():
        camera.open(0)
    return "Prone_elbow Camera started"

@app.route('/Prone_elbow/video_feed')
def video_feed_Prone_elbow():
    with open('Prone on elbow6.pkl', 'rb') as f:
        model = pickle.load(f)
    display_text = "TIME"
    return Response(generate_frames(model,display_text), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Prone_elbow/stop_video')
def stop_video_Prone_elbow():
    global camera
    if camera.isOpened():
        camera.release()
    return "Prone_elbow Camera stopped"

@app.route('/Prone_press_up') #Prone press up
def Prone_press_up():
    return render_template('Press up.html')

@app.route('/Prone_press_up/start_video')
def start_video_Prone_press_up():
    global camera
    if not camera.isOpened():
        camera.open(0)
    return "Prone_press_up Camera started"

@app.route('/Prone_press_up/video_feed')
def video_feed_Prone_press_up():
    with open('Prone press up04.pkl', 'rb') as f:
        model = pickle.load(f)
    display_text = "REPS"
    return Response(generate_frames(model,display_text), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Prone_press_up/stop_video')
def stop_video_Prone_press_up():
    global camera
    if camera.isOpened():
        camera.release()
    return "Prone_press_up Camera stopped"

@app.route('/Standing_extension') #Standing extension
def Standing_extension():
    return render_template('Standing.html')

@app.route('/Standing_extension/start_video')
def start_video_Standing_extension():
    global camera
    if not camera.isOpened():
        camera.open(0)
    return "Standing_extension Camera started"

@app.route('/Standing_extension/video_feed')
def video_feed_Standing_extension():
    return Response(generate_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Standing_extension/stop_video')
def stop_video_Standing_extension():
    global camera
    if camera.isOpened():
        camera.release()
    return "Standing_extension Camera stopped"

@app.route('/Lying_flexion') #Lying flexion
def Lying_flexion():
    return render_template('Lying.html')

@app.route('/Lying_flexion/start_video')
def start_video_Lying_flexion():
    global camera
    if not camera.isOpened():
        camera.open(0)
    return "Lying_flexion Camera started"

@app.route('/Lying_flexion/video_feed')
def video_feed_Lying_flexion():
    with open('Lying_Flexion.pkl', 'rb') as f:
        model = pickle.load(f)
    display_text = "REPS"
    return Response(generate_frames(model,display_text), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Lying_flexion/stop_video')
def stop_video_Lying_flexion():
    global camera
    if camera.isOpened():
        camera.release()
    return "lying_flexion Camera stopped"

@app.route('/Sitting_flexion') #Sitting flexion
def Sitting_flexion():
    return render_template('SittingFlexion.html')

@app.route('/Sitting_flexion/start_video')
def start_video_Sitting_flexion():
    global camera
    if not camera.isOpened():
        camera.open(0)
    return "Sitting_flexion Camera started"

@app.route('/Sitting_flexion/video_feed')
def video_feed_Sitting_flexion():
    with open('SittingFlexion.pkl', 'rb') as f:
        model = pickle.load(f)
    display_text = "REPS"
    return Response(generate_frames(model,display_text), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Sitting_flexion/stop_video')
def stop_video_Sitting_flexion():
    global camera
    if camera.isOpened():
        camera.release()
    return "Sitting_flexion Camera stopped"

@app.route('/Standing_flexion') #Standing flexion
def Standing_flexion():
    return render_template('StandingFlexion.html')

@app.route('/Standing_flexion/start_video')
def start_video_Standing_flexion():
    global camera
    if not camera.isOpened():
        camera.open(0)
    return "Standing_flexion Camera started"

@app.route('/Standing_flexion/video_feed')
def video_feed_Standing_flexion():
    with open('StandingFlexion.pkl', 'rb') as f:
        model = pickle.load(f)
    display_text = "REPS"
    return Response(generate_frames(model,display_text), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Standing_flexion/stop_video')
def stop_video_Standing_flexion():
    global camera
    if camera.isOpened():
        camera.release()
    return "Standing_flexion Camera stopped"

if __name__ == '__main__':
    app.run()
