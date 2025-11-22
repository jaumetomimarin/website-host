import cv2
import threading
from flask import Flask, Response

app = Flask(__name__)

# Inicializar la cámara y el substractor de fondo
cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2(200, 16)

def generate_frames(stream_type):
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            if stream_type == 'real':
                # Para el stream real, simplemente codificamos el frame
                ret, buffer = cv2.imencode('.jpg', frame)
            elif stream_type == 'virtual':
                # Para el stream virtual, aplicamos el procesamiento
                fg_mask = backSub.apply(frame, learningRate=0.7)
                retval, mask_thresh = cv2.threshold(fg_mask, 120, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
                contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_contour_area = 500
                large_contours1 = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
                ret, buffer = cv2.imencode('.jpg', mask_eroded)

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed_real')
def video_feed_real():
    return Response(generate_frames('real'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_virtual')
def video_feed_virtual():
    return Response(generate_frames('virtual'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)