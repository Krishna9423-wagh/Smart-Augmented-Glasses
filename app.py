import cv2
import sqlite3
import numpy as np
import os
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# ── Face detector ──────────────────────────────────────────────
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ── LBPH Recognizer (OpenCV 5 compatible) ─────────────────────
recognizer = None
MODEL_PATH = os.path.join('recognizer', 'trainingdata.yml')
if os.path.exists(MODEL_PATH):
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)
        print("✅ Recognition model loaded successfully.")
    except Exception as e:
        recognizer = None
        print(f"⚠️  Could not load model (incompatible format): {e}")
        print("   Face detection will still work. Re-train with dataSet_trainer.py to enable recognition.")

# ── Database lookup ────────────────────────────────────────────
def getProfile(face_id):
    try:
        conn = sqlite3.connect("facedatabase.db")
        cursor = conn.execute("SELECT * FROM details WHERE id=?", (face_id,))
        profile = None
        for row in cursor:
            profile = row
        conn.close()
        return profile
    except Exception:
        return None

# ── Detection state shared across frames ───────────────────────
detection_state = {"faces": [], "names": []}

# ── Video stream generator ─────────────────────────────────────
camera = None

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera

def generate_frames():
    cam = get_camera()
    while True:
        success, frame = cam.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        detected_names = []
        for (x, y, w, h) in faces:
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)

            name = "Unknown"
            confidence_text = ""

            if recognizer is not None:
                face_id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                if conf < 80:
                    profile = getProfile(face_id)
                    if profile:
                        name = str(profile[1])
                        confidence_text = f"{round(100 - conf)}%"
                    else:
                        name = f"ID:{face_id}"
                else:
                    name = "Unknown"

            detected_names.append(name)

            # Background label
            label = f"{name} {confidence_text}".strip()
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
            cv2.rectangle(frame, (x, y - th - 12), (x + tw + 10, y), (0, 200, 255), -1)
            cv2.putText(frame, label, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)

        detection_state["faces"] = len(faces)
        detection_state["names"] = detected_names

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ── Routes ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({
        "faces_detected": detection_state["faces"],
        "names": detection_state["names"],
        "model_loaded": recognizer is not None
    })


if __name__ == '__main__':
    print("\n🕶️  Smart Augmented Glasses — Web Interface")
    print("=" * 45)
    print("🌐  Open your browser at: http://localhost:8080")
    print("=" * 45 + "\n")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
