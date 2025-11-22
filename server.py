import cv2
import threading
from flask import Flask, Response, send_from_directory
import os
import numpy as np

app = Flask(__name__)

# Servir archivos estáticos
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

# Inicializar la cámara - en Render usaremos un video de prueba
try:
    # En Render no hay cámara física, usamos video de prueba
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Si no funciona la cámara, usar video de prueba
        cap = cv2.VideoCapture('https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4')
except:
    # Video de prueba como fallback
    cap = cv2.VideoCapture('https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4')

backSub = cv2.createBackgroundSubtractorMOG2(200, 16)

def generate_frames(stream_type):
    while True:
        try:
            success, frame = cap.read()
            if not success:
                # Reiniciar video si llega al final
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
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
        except Exception as e:
            # En caso de error, generar frame de error
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, f"Error: {str(e)}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)