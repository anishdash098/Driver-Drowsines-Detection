import os
import time
import urllib.request
import cv2
import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from collections import deque

# 1. Initialize Alarm Sound
pygame.mixer.init()
alarm_path = "alarm.wav"
alarm_sound = pygame.mixer.Sound(alarm_path) if os.path.exists(alarm_path) else None

# Helper: Version-proof model loader
def load_safe_model(model_path, num_classes=2):
    try:
        return load_model(model_path)
    except Exception:
        base_model = MobileNetV2(weights=None, include_top=False, input_shape=(128, 128, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)
        model.load_weights(model_path)
        return model

# 2. Load Trained MobileNetV2 Models
print("[INFO] Loading Models...")
eye_model = load_safe_model("mobilenet_eye_model.h5", num_classes=2)
yawn_model = load_safe_model("mobilenet_yawn_model.h5", num_classes=2)
print("✅ Models loaded successfully!")

# 3. Load Cascades for Face & Eyes
ClassifierClass = getattr(cv2, 'CascadeClassifier', None)
if ClassifierClass is None and hasattr(cv2, 'objdetect'):
    ClassifierClass = getattr(cv2.objdetect, 'CascadeClassifier', None)

cascade_dir = getattr(getattr(cv2, 'data', None), 'haarcascades', '')

def get_cascade(name, url):
    p = os.path.join(cascade_dir, name)
    if not os.path.exists(p):
        p = name
        if not os.path.exists(p):
            urllib.request.urlretrieve(url, p)
    return ClassifierClass(p)

face_cascade = get_cascade(
    "haarcascade_frontalface_default.xml",
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
)
eye_cascade = get_cascade(
    "haarcascade_eye.xml",
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
)

# 4. Open Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

# Thresholds
EYE_ALERT_THRESHOLD = 15   # ~1.5s of closed eyes
YAWN_ALERT_THRESHOLD = 14  # ~1.5s of sustained yawning

eye_score = 0
yawn_score = 0
alarm_active = False

eye_conf_history = deque(maxlen=4)
yawn_conf_history = deque(maxlen=4)

prev_time = time.time()
window_name = "Driver Drowsiness Monitoring System"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("\n[INFO] Compact HUD & Enhanced Yawn System Running!\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time + 1e-6))
        prev_time = curr_time

        height, width, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

        current_eye_state = "No Face"
        current_yawn_state = "No Face"
        eye_display_conf = 0.0
        yawn_display_conf = 0.0
        eye_preds = []

        for (x, y, w, h) in faces:
            # Subtle face box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 120, 0), 1)

            # ----------------------------------------------------
            # 1. EYE DETECTION (HIGH ACCURACY)
            # ----------------------------------------------------
            roi_face_gray = gray[y : y + int(h * 0.6), x : x + w]
            roi_face_color = frame[y : y + int(h * 0.6), x : x + w]

            detected_eyes = eye_cascade.detectMultiScale(
                roi_face_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(25, 25)
            )

            if len(detected_eyes) > 0:
                for (ex, ey, ew, eh) in detected_eyes:
                    cv2.rectangle(roi_face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
                    eye_crop = roi_face_color[ey : ey + eh, ex : ex + ew]
                    if eye_crop.size > 0:
                        eye_img = cv2.resize(eye_crop, (128, 128))
                        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
                        eye_batch = preprocess_input(np.expand_dims(eye_img.astype('float32'), axis=0))
                        pred = eye_model.predict(eye_batch, verbose=0)[0]
                        eye_preds.append(pred)
            else:
                eye_w, eye_h = int(w * 0.3), int(h * 0.25)
                left_eye = frame[y + int(h*0.2) : y + int(h*0.2) + eye_h, x + int(w*0.15) : x + int(w*0.15) + eye_w]
                right_eye = frame[y + int(h*0.2) : y + int(h*0.2) + eye_h, x + int(w*0.55) : x + int(w*0.55) + eye_w]
                
                for zone in [left_eye, right_eye]:
                    if zone.size > 0:
                        eye_img = cv2.resize(zone, (128, 128))
                        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
                        eye_batch = preprocess_input(np.expand_dims(eye_img.astype('float32'), axis=0))
                        pred = eye_model.predict(eye_batch, verbose=0)[0]
                        eye_preds.append(pred)

            if len(eye_preds) > 0:
                avg_closed_prob = np.mean([p[0] for p in eye_preds])
                avg_open_prob   = np.mean([p[1] for p in eye_preds])

                if avg_closed_prob > 0.50:
                    current_eye_state = "Closed"
                    eye_display_conf = avg_closed_prob * 100
                else:
                    current_eye_state = "Open"
                    eye_display_conf = avg_open_prob * 100

                eye_conf_history.append(eye_display_conf)
                eye_display_conf = np.mean(eye_conf_history)

            # ----------------------------------------------------
            # 2. RE-CALIBRATED ROBUST YAWN DETECTION
            # ----------------------------------------------------
            # Focus strictly on lip opening aperture (y + 0.65h to y + 0.90h)
            my1 = max(0, y + int(h * 0.65))
            my2 = min(height, y + int(h * 0.90))
            mx1 = max(0, x + int(w * 0.22))
            mx2 = min(width, x + int(w * 0.78))
            mouth_roi_gray = gray[my1:my2, mx1:mx2]
            mouth_roi_color = frame[my1:my2, mx1:mx2]

            if mouth_roi_gray.size > 0:
                # 1. MobileNet Yawn Model
                mouth_img = cv2.resize(mouth_roi_color, (128, 128))
                mouth_img = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2RGB)
                mouth_batch = preprocess_input(np.expand_dims(mouth_img.astype('float32'), axis=0))
                yawn_probs = yawn_model.predict(mouth_batch, verbose=0)[0]
                prob_no_yawn, prob_yawn = yawn_probs[0], yawn_probs[1]

                # 2. Physical Vertical Aperture Ratio
                # Detect inner mouth darkness relative to surrounding lips/beard
                blurred_mouth = cv2.GaussianBlur(mouth_roi_gray, (7, 7), 0)
                mean_brightness = np.mean(blurred_mouth)
                # When mouth opens, inner cavity is darker than the surrounding skin/beard
                dark_threshold = max(20, mean_brightness - 35)
                _, thresh = cv2.threshold(blurred_mouth, dark_threshold, 255, cv2.THRESH_BINARY_INV)
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                yawn_opening_ratio = 0.0

                for c in contours:
                    cx, cy, cw, ch = cv2.boundingRect(c)
                    # Yawn opening must be sufficiently wide and tall
                    if cw > int(w * 0.12) and ch > int(h * 0.08):
                        yawn_opening_ratio = max(yawn_opening_ratio, float(ch) / float(cw))

                # True Yawn:
                # Either prominent physical cavity opening (> 0.50) OR high-confidence open mouth
                if (yawn_opening_ratio > 0.45 and prob_yawn > 0.60) or (prob_yawn > 0.88 and yawn_opening_ratio > 0.30):
                    current_yawn_state = 'Yawning'
                    yawn_display_conf = prob_yawn * 100
                    box_color = (0, 0, 255) # Red box on yawn
                else:
                    current_yawn_state = 'Normal'
                    yawn_display_conf = prob_no_yawn * 100
                    box_color = (0, 255, 0) # Green box on normal

                yawn_conf_history.append(yawn_display_conf)
                yawn_display_conf = np.mean(yawn_conf_history)
                cv2.rectangle(frame, (mx1, my1), (mx2, my2), box_color, 1)

        # ----------------------------------------------------
        # SCORES & ALARM
        # ----------------------------------------------------
        if current_eye_state == 'Closed':
            eye_score = min(30, eye_score + 1)
        else:
            eye_score = max(0, eye_score - 2)

        if current_yawn_state == 'Yawning':
            yawn_score = min(30, yawn_score + 1)
        else:
            yawn_score = max(0, yawn_score - 2)

        # Alarm Logic
        if eye_score >= EYE_ALERT_THRESHOLD:
            cv2.putText(frame, "DROWSINESS ALERT!", (width//2 - 180, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            if alarm_sound and not alarm_active:
                alarm_sound.play(-1)
                alarm_active = True
        else:
            if alarm_sound and alarm_active:
                alarm_sound.stop()
                alarm_active = False

        if yawn_score >= YAWN_ALERT_THRESHOLD:
            cv2.putText(frame, "FATIGUE WARNING: YAWNING!", (width//2 - 210, height // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # ----------------------------------------------------
        # COMPACT SIDEBAR PANEL (TOP-LEFT)
        # ----------------------------------------------------
        panel_w, panel_h = 320, 105
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (15, 15), (15 + panel_w, 15 + panel_h), (70, 70, 70), 1)

        # Eyes row
        eye_color = (0, 255, 0) if current_eye_state == 'Open' else (0, 0, 255)
        cv2.putText(frame, f"Eyes : {current_eye_state}", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, eye_color, 1)
        cv2.putText(frame, f"{eye_display_conf:.1f}%", (160, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
        cv2.putText(frame, f"[{eye_score}/{EYE_ALERT_THRESHOLD}]", (245, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        # Mouth row
        yawn_color = (0, 255, 0) if current_yawn_state == 'Normal' else (0, 165, 255)
        cv2.putText(frame, f"Mouth: {current_yawn_state}", (30, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.52, yawn_color, 1)
        cv2.putText(frame, f"{yawn_display_conf:.1f}%", (160, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
        cv2.putText(frame, f"[{yawn_score}/{YAWN_ALERT_THRESHOLD}]", (245, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        # Footer row: FPS & Model
        cv2.putText(frame, f"FPS: {fps}  |  MobileNetV2", (30, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

except KeyboardInterrupt:
    pass

finally:
    if alarm_sound:
        alarm_sound.stop()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
