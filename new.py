from flask import Flask, render_template, send_from_directory, Response, request, redirect, url_for, flash
from ultralytics import YOLO
from playsound import playsound
import cv2
import os
import threading

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"  # For flash messages

# Load the trained YOLO model
model = YOLO(r"C:\Users\gamer\PycharmProjects\project\best.pt")

# Configure directories
UPLOAD_FOLDER = r"C:\Users\gamer\PycharmProjects\project\uploads"
OUTPUT_FOLDER = r"C:\Users\gamer\PycharmProjects\project\outputs"
STATIC_FOLDER = r"C:\Users\gamer\PycharmProjects\project\static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["STATIC_FOLDER"] = STATIC_FOLDER

# Initialize webcam
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Mapping of detected signs to sound files
audio_files = {
    "AWAK": r"C:\Users\gamer\PycharmProjects\project\static\awak.mp3",
    "MAAF": r"C:\Users\gamer\PycharmProjects\project\static\maaf.mp3",
    "MAKAN": r"C:\Users\gamer\PycharmProjects\project\static\makan.mp3",
    "MINUM": r"C:\Users\gamer\PycharmProjects\project\static\minum.mp3",
    "SALAH": r"C:\Users\gamer\PycharmProjects\project\static\salah.mp3",
    "SAYA": r"C:\Users\gamer\PycharmProjects\project\static\saya.mp3",
    "TOLONG": r"C:\Users\gamer\PycharmProjects\project\static\tolong.mp3",
}

# Global variable to track if sound is playing
sound_playing = False


def play_sound(detected_sign):
    """Play the corresponding audio file for a detected sign."""
    global sound_playing
    if sound_playing:
        return  # Prevent playing a new sound if one is already playing

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
    """Render the result page."""
    detected_sign = request.args.get("detected_sign", "No Sign Detected")
    is_video = filename.lower().endswith(".mp4")

    # Construct the relative URL for the output file
    file_url = f"/outputs/{filename}"

    # Render the result page
    return render_template("result.html", detected_sign=detected_sign, file_url=file_url, is_video=is_video)


@app.route("/outputs/<path:filename>")
def outputs_static(filename):
    """Serve files from outputs folder as static files."""
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def generate_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break

            results = model.predict(frame, conf=0.5, show=False)
            annotated_frame = results[0].plot()

            # Extract detected class indices and map to labels
            detected_signs = []
            for box in results[0].boxes.data.tolist():
                class_idx = int(box[5])  # Extract class index
                if class_idx < len(model.names):
                    detected_signs.append(model.names[class_idx])

            # Play sound for the first detected sign
            if detected_signs:
                detected_sign = detected_signs[0]
                print(f"Dikesan: {detected_sign}")
                threading.Thread(target=play_sound, args=(detected_sign,)).start()

            _, buffer = cv2.imencode(".jpg", annotated_frame)
            frame = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


def save_annotated_file(results, output_filename, is_video=False, input_path=None):
    """Saves annotated images or videos and extracts detected signs."""
    detected_signs = []

    if is_video:
        # Ensure consistent naming for output video
        output_filename = output_filename.replace(".mp4", "_annotated.mp4")
        cap = cv2.VideoCapture(input_path)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24  # Default FPS fallback
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec for .mp4 files
        out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=0.65, show=False)
            annotated_frame = results[0].plot()

            # Resize frame if necessary
            if annotated_frame.shape[1] != frame_width or annotated_frame.shape[0] != frame_height:
                annotated_frame = cv2.resize(annotated_frame, (frame_width, frame_height))

            out.write(annotated_frame)

            # Extract detected signs
            for box in results[0].boxes.data.tolist():
                class_idx = int(box[5])
                if class_idx < len(model.names):
                    detected_signs.append(model.names[class_idx])

        cap.release()
        out.release()
    else:
        output_filename = output_filename.replace(".jpg", "_annotated.png")
        annotated_image = results[0].plot()
        cv2.imwrite(output_filename, annotated_image)

        # Extract detected signs
        for box in results[0].boxes.data.tolist():
            class_idx = int(box[5])
            if class_idx < len(model.names):
                detected_signs.append(model.names[class_idx])

    return detected_signs[0] if detected_signs else "No Sign Detected", output_filename


@app.route("/predict_video", methods=["POST"])
def predict_video():
    """Handle video upload and prediction."""
    if "video" not in request.files:
        flash("No video uploaded.")
        return redirect(url_for("home"))

    file = request.files["video"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("home"))

    input_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(input_path)

    # Generate annotated video filename
    base_filename = os.path.splitext(file.filename)[0]
    output_filename = os.path.join(app.config["OUTPUT_FOLDER"], f"{base_filename}.mp4")

    detected_sign, final_filename = save_annotated_file(None, output_filename, is_video=True, input_path=input_path)
    return redirect(url_for("result", filename=os.path.basename(final_filename), detected_sign=detected_sign, is_video=True))


@app.route("/predict_image", methods=["POST"])
def predict_image():
    """Handle image upload and prediction."""
    if "image" not in request.files:
        flash("No image uploaded.")
        return redirect(url_for("home"))

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("home"))

    input_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(input_path)

    output_filename = os.path.join(app.config["OUTPUT_FOLDER"], f"annotated_{os.path.splitext(file.filename)[0]}.png")
    results = model.predict(source=input_path, conf=0.7)

    detected_sign = save_annotated_file(results, output_filename)
    return redirect(url_for("result", filename=os.path.basename(output_filename), detected_sign=detected_sign, is_video=False))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
