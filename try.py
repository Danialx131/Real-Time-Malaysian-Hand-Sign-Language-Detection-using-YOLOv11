from flask import Flask, render_template, send_from_directory, Response, request, redirect, url_for, flash
from ultralytics import YOLO
from playsound import playsound
import cv2
import numpy as np
import os
import threading

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"  # For flash messages

# Load the trained YOLO model
model = YOLO(r"D:\project\Real Time Malaysian Sign Language Detection Using YoloV11\Real Time Malaysian Sign Language Detection Using YoloV11 - project\roboflowYolov12.pt")

# Configure directories
UPLOAD_FOLDER = r"D:\project\Real Time Malaysian Sign Language Detection Using YoloV11\Real Time Malaysian Sign Language Detection Using YoloV11 - project\uploads"
OUTPUT_FOLDER = r"D:\project\Real Time Malaysian Sign Language Detection Using YoloV11\Real Time Malaysian Sign Language Detection Using YoloV11 - project\outputs"
STATIC_FOLDER = r"D:\project\Real Time Malaysian Sign Language Detection Using YoloV11\Real Time Malaysian Sign Language Detection Using YoloV11 - project\static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["STATIC_FOLDER"] = STATIC_FOLDER

# Initialize webcam
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("❌ Could not open webcam. Is it busy or not present?")
else:
    print("✅ Webcam opened successfully.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Mapping of detected signs to sound files
audio_files = {
    "AWAK": r"D:\project\Real Time Malaysian Sign Language Detection Using YoloV11\Real Time Malaysian Sign Language Detection Using YoloV11 - project\static\awak.mp3",
    "MAAF": r"D:\project\Real Time Malaysian Sign Language Detection Using YoloV11\Real Time Malaysian Sign Language Detection Using YoloV11 - project\static\Maaf.mp3",
    "MAKAN": r"D:\project\Real Time Malaysian Sign Language Detection Using YoloV11\Real Time Malaysian Sign Language Detection Using YoloV11 - project\static\makan.mp3",
    "MINUM": r"D:\project\Real Time Malaysian Sign Language Detection Using YoloV11\Real Time Malaysian Sign Language Detection Using YoloV11 - project\static\minum.mp3",
    "SALAH": r"D:\project\Real Time Malaysian Sign Language Detection Using YoloV11\Real Time Malaysian Sign Language Detection Using YoloV11 - project\static\salah.mp3",
    "SAYA": r"D:\project\Real Time Malaysian Sign Language Detection Using YoloV11\Real Time Malaysian Sign Language Detection Using YoloV11 - project\static\saya.mp3",
    "TOLONG": r"D:\project\Real Time Malaysian Sign Language Detection Using YoloV11\Real Time Malaysian Sign Language Detection Using YoloV11 - project\static\tolong.mp3",
}

# Global variable to track if sound is playing
sound_playing = False

def play_sound(detected_sign):
    global sound_playing
    if sound_playing:
        return
    sound_playing = True
    audio_path = audio_files.get(detected_sign)
    if audio_path:
        try:
            playsound(audio_path)
        except Exception as e:
            print(f"Error playing sound for {detected_sign}: {e}")
    sound_playing = False

@app.route("/result/<path:filename>")
def result(filename):
    detected_sign = request.args.get("detected_sign", "No Sign Detected")
    is_video = filename.lower().endswith(".mp4")
    file_url = f"/outputs/{filename}"
    return render_template("result.html", detected_sign=detected_sign, file_url=file_url, is_video=is_video)

@app.route("/outputs/<path:filename>")
def outputs_static(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    print(">> /video_feed endpoint called")
    def generate_frames():
        print(">> generate_frames called")
        while True:
            success, frame = camera.read()
            if not success:
                print("❌ Failed to grab frame from webcam.")
                blank_frame = 255 * np.ones((640, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode(".jpg", blank_frame)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
                continue
            else:
                print("✅ Frame grabbed successfully.")

            try:
                results = model.predict(frame, conf=0.5, show=False)
                annotated_frame = results[0].plot()
            except Exception as e:
                print(f"⚠️ YOLO prediction error: {e}")
                annotated_frame = frame

            try:
                _, buffer = cv2.imencode(".jpg", annotated_frame)
                frame_bytes = buffer.tobytes()
            except Exception as e:
                print(f"⚠️ JPEG encoding error: {e}")
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# The rest of your code for save_annotated_file, predict_video, predict_image remains unchanged...

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
