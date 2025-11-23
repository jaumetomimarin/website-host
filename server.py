import cv2
import threading
from flask import Flask, Response
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)

# Variables globals per als frames
latest_frame_real = None
latest_frame_virtual = None
frame_lock = threading.Lock()
camera_active = False

def camera_thread():
    global latest_frame_real, latest_frame_virtual, camera_active
    
    cap = cv2.VideoCapture(0)
    backSub = cv2.createBackgroundSubtractorMOG2(200, 16)
    camera_active = True
    
    while camera_active:
        success, frame = cap.read()
        if success:
            # Frame real
            ret_real, buffer_real = cv2.imencode('.jpg', frame)
            frame_real = buffer_real.tobytes()
            
            # Frame virtual
            fg_mask = backSub.apply(frame, learningRate=0.7)
            retval, mask_thresh = cv2.threshold(fg_mask, 120, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
            ret_virtual, buffer_virtual = cv2.imencode('.jpg', mask_eroded)
            frame_virtual = buffer_virtual.tobytes()
            
            with frame_lock:
                latest_frame_real = frame_real
                latest_frame_virtual = frame_virtual
        
        time.sleep(0.03)
    
    cap.release()

def generate_frames(stream_type):
    global camera_active
    
    # Iniciar càmera si no està activa
    if not camera_active:
        thread = threading.Thread(target=camera_thread)
        thread.daemon = True
        thread.start()
        # Esperar a que la càmera comenci
        time.sleep(1)
    
    while True:
        with frame_lock:
            if stream_type == 'real' and latest_frame_real:
                frame_bytes = latest_frame_real
            elif stream_type == 'virtual' and latest_frame_virtual:
                frame_bytes = latest_frame_virtual
            else:
                frame_bytes = None
        
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Frame buit si no hi ha dades
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
        
        time.sleep(0.03)

@app.route('/video_feed_real')
def video_feed_real():
    return Response(generate_frames('real'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_virtual')
def video_feed_virtual():
    return Response(generate_frames('virtual'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)