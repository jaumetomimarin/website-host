import cv2
import threading
from flask import Flask, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---- Cámara compartida ----
cap = cv2.VideoCapture(0)

last_frame = None
backSub = cv2.createBackgroundSubtractorMOG2(200, 16)

def camera_thread():
    global last_frame
    while True:
        ok, frame = cap.read()
        if ok:
            last_frame = frame

# Iniciar hilo de captura único
threading.Thread(target=camera_thread, daemon=True).start()


# ---- STREAM REAL ----
def generate_frames_real():
    global last_frame

    while True:
        if last_frame is None:
            continue

        ret, buffer = cv2.imencode(".jpg", last_frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


# ---- STREAM VIRTUAL (procesado) ----
def generate_frames_virtual():
    global last_frame

    while True:
        if last_frame is None:
            continue

        fg_mask = backSub.apply(last_frame, learningRate=0.7)
        retval, mask_thresh = cv2.threshold(fg_mask, 120, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

        ret, buffer = cv2.imencode('.jpg', mask_eroded)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/video_feed_real")
def video_feed_real():
    return Response(generate_frames_real(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed_virtual")
def video_feed_virtual():
    return Response(generate_frames_virtual(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
