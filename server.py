import cv2
import threading
from flask import Flask, Response
from flask_cors import CORS
import numpy as np
import time
import open3d as o3d
import mediapipe as mp
import copy

# ==========================================
# 1. CONFIGURACIÓN Y VARIABLES GLOBALES
# ==========================================

app = Flask(__name__)
CORS(app)

# Parámetros de Cámara
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

# Parámetros de Reconstrucción 3D
BONE_THICKNESS = 0.05 
BONE_COLOR = [1.0, 0.0, 0.0]
AXIS_CORRECTION = np.array([-1.0, -1.0, 1.0]) 
MEDIAPIPE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27),
    (24, 26), (26, 28), (11, 12), (23, 24), (11, 23), (12, 24),
    (0, 11), (0, 12)
]

# Variables Compartidas
latest_virtual_frame = None  
frame_lock = threading.Lock() 

# Referencias globales para los hilos de las cámaras (se inicializan en __main__)
camera_front = None 
camera_side = None

# ==========================================
# 2. GESTORES DE THREADS DE CÁMARA (SOLUCIÓN AL CONFLICTO)
# ==========================================

class CameraStream(threading.Thread):
    """
    Clase para gestionar la captura de un solo cv2.VideoCapture en un hilo separado
    y compartir el último frame de forma segura.
    """
    def __init__(self, index):
        threading.Thread.__init__(self)
        self.index = index
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
        self.latest_frame = None
        self.lock = threading.Lock() 
        self.running = True
        print(f"Cámara {index} abierta: {self.cap.isOpened()}")

    def run(self):
        while self.running:
            # Captura el frame
            success, frame = self.cap.read()
            
            if success:
                # Almacena el frame en el buffer compartido
                with self.lock:
                    self.latest_frame = frame
            else:
                # Muestra un frame de error si la cámara falla
                error_frame = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"CAM {self.index} ERROR", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                with self.lock:
                    self.latest_frame = error_frame.copy()

            # Tasa de frames para la captura
            time.sleep(1/60) 
            
    def get_frame(self):
        """Devuelve una copia del frame más reciente capturado."""
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()
            
    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
            
# ==========================================
# 3. FUNCIONES DE AYUDA Y CLASE DE PROCESAMIENTO 3D
# ==========================================

def setup_orthogonal_cameras():
    """Retorna las matrices de cámara (intrínsecas y extrínsecas)."""
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
    R1 = np.eye(3, dtype=np.float64); t1 = np.zeros((3,1), dtype=np.float64)
    c, s = np.cos(np.pi/2), np.sin(np.pi/2) 
    R2 = np.array([[ c, 0, s], [ 0, 1, s], [-s, 0, c]], dtype=np.float64)
    C2 = np.array([3.0, 0.0, 4.0], dtype=np.float64).reshape(3,1) 
    t2 = -R2 @ C2
    return (K, R1, t1), (K, R2, t2)

def extract_mediapipe_data(image_bgr, detector):
    """Procesa una imagen para obtener landmarks 2D y 3D (world)."""
    if image_bgr is None: return None, None, None
    h, w, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = detector.process(image_rgb)
    if not results.pose_landmarks: return None, None, None
    lm_2d = np.array([[lm.x * w, lm.y * h] for lm in results.pose_landmarks.landmark], dtype=np.float64)
    lm_3d_mp = []; confidence = []
    for lm in results.pose_world_landmarks.landmark:
        lm_3d_mp.append([lm.x, lm.y, lm.z]); confidence.append(lm.visibility)
    return lm_2d, np.array(lm_3d_mp), np.array(confidence)

def triangulate_landmarks(lm1, lm2, cam1, cam2):
    """Triangula landmarks 2D de dos cámaras a 3D."""
    K1, R1, t1 = cam1; K2, R2, t2 = cam2
    P1 = K1 @ np.hstack((R1, t1.reshape(3,1))); P2 = K2 @ np.hstack((R2, t2.reshape(3,1)))
    Xh = cv2.triangulatePoints(P1, P2, lm1.T, lm2.T)
    div = Xh[3]; div[np.abs(div) < 1e-6] = 1e-6 
    X = (Xh[:3] / div).T
    return X

def fuse_data(mp_3d, tri_3d, conf):
    """Alinea y fusiona los datos de Mediapipe 3D con la triangulación."""
    if len(mp_3d) == 0 or len(tri_3d) == 0: return mp_3d
    valid = ~np.isnan(tri_3d).any(axis=1) & ~np.isnan(mp_3d).any(axis=1)
    tri_aligned = tri_3d.copy()
    if valid.sum() > 5:
        try:
            src = tri_3d[valid]; dst = mp_3d[valid]
            src_mean = src.mean(axis=0); dst_mean = dst.mean(axis=0)
            cov = ((dst - dst_mean).T @ (src - src_mean)) / src.shape[0]
            U, S, Vt = np.linalg.svd(cov); R = U @ Vt
            if np.linalg.det(R) < 0: Vt[2,:] *= -1; R = U @ Vt
            t = dst_mean - (R @ src_mean); tri_aligned = (tri_3d @ R.T) + t
        except np.linalg.LinAlgError: pass
    w = np.clip(conf, 0, 1)[:, None]; fused = w * mp_3d + (1 - w) * tri_aligned
    return fused

def create_cylinder_mesh(p0, p1, radius=0.05, color=[0, 1, 0]):
    """Crea una malla de cilindro para simular un hueso."""
    p0 = np.array(p0); p1 = np.array(p1); v = p1 - p0
    length = np.linalg.norm(v)
    if length < 1e-6: return None
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=8)
    cylinder.paint_uniform_color(color)
    z_axis = np.array([0, 0, 1]); v_normalized = v / length; axis = np.cross(z_axis, v_normalized); axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6: R = np.diag([1, -1, -1]) if np.dot(z_axis, v_normalized) < 0 else np.eye(3)
    else:
        axis = axis / axis_len; angle = np.arccos(np.clip(np.dot(z_axis, v_normalized), -1.0, 1.0))
        R = cylinder.get_rotation_matrix_from_axis_angle(axis * angle)
    cylinder.rotate(R, center=np.array([0, 0, 0])); cylinder.translate((p0 + p1) / 2)
    return cylinder


# Assegura't de tenir imports i variables globals iguals que abans
HEAD_OBJ_PATH = "dummy_gorgon.obj" 
HEAD_SCALE = 0.05 

class ReconstructionProcessor(threading.Thread):
    def __init__(self, camera_front_stream, camera_side_stream, frame_lock):
        threading.Thread.__init__(self)
        self.camera_front_stream = camera_front_stream
        self.camera_side_stream = camera_side_stream
        self.frame_lock = frame_lock
        self.running = True

    def run(self):
        global latest_virtual_frame
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Reconstruction", width=1200, height=600, visible=False) 
        vis.get_render_option().background_color = np.array([0,0,0])
        
        # 1. Càmera virtual (Zoom i posició)
        ctr = vis.get_view_control()
        ctr.set_zoom(0.6)
        ctr.set_lookat([0, 0, 0])
        ctr.set_front([0, 0, -1])  # Mira des del davant
        ctr.set_up([0, -1, 0])     # Y invertida (habitual en CV)

        # 2. Inicialització de Geometries
        skeleton_mesh = o3d.geometry.TriangleMesh()
        vis.add_geometry(skeleton_mesh)
        
        # Carreguem el cap (NOMÉS UNA VEGADA)
        try:
            head_base = o3d.io.read_triangle_mesh(HEAD_OBJ_PATH)
            head_base.compute_vertex_normals()
            head_base.paint_uniform_color([0.9, 0.8, 0.7])
            head_base.scale(HEAD_SCALE, center=[0,0,0]) # Escalem aquí una sola vegada
            # Si el teu OBJ mira cap enrere, descomenta això:
            # head_base.rotate(head_base.get_rotation_matrix_from_xyz((0, np.pi, 0)), center=(0,0,0))
        except:
            head_base = None
            print("No s'ha trobat head.obj")

        head_active = o3d.geometry.TriangleMesh()
        vis.add_geometry(head_active)

        # Inicialització MP (igual que abans)
        mp_pose = mp.solutions.pose
        pose_front = mp_pose.Pose(min_detection_confidence=0.5, model_complexity=1)
        pose_side = mp_pose.Pose(min_detection_confidence=0.5, model_complexity=1)
        cam1_params, cam2_params = setup_orthogonal_cameras()

        while self.running:
            frame1 = self.camera_front_stream.get_frame() 
            frame2 = self.camera_side_stream.get_frame() 
            if frame1 is None or frame2 is None: time.sleep(0.01); continue

            # ... (Processament MP i Triangulació igual que abans) ...
            p2d_f, p3d_mp_f, conf_f = extract_mediapipe_data(frame1, pose_front)
            p2d_s, p3d_mp_s, conf_s = extract_mediapipe_data(frame2, pose_side)
            
            final_landmarks = None
            # ... (Bloc de fusió igual que abans) ...
            if p2d_f is not None and p2d_s is not None:
                try:
                    tri_3d = triangulate_landmarks(p2d_f, p2d_s, cam1_params, cam2_params)
                    fused = fuse_data(p3d_mp_f, tri_3d, (conf_f + conf_s)/2)
                    final_landmarks = fused * AXIS_CORRECTION
                except: pass

            # 3. ACTUALITZACIÓ DE GEOMETRIA
            if final_landmarks is not None:
                # A. ESQUELET (Igual que abans)
                vis.remove_geometry(skeleton_mesh, reset_bounding_box=False)
                skeleton_mesh = o3d.geometry.TriangleMesh()
                for i, j in MEDIAPIPE_CONNECTIONS:
                    if i < len(final_landmarks) and j < len(final_landmarks):
                        cyl = create_cylinder_mesh(final_landmarks[i], final_landmarks[j], BONE_THICKNESS, BONE_COLOR)
                        if cyl: skeleton_mesh += cyl
                vis.add_geometry(skeleton_mesh, reset_bounding_box=False)

                # B. CAP (SIMPLIFICAT)
                if head_base is not None:
                    vis.remove_geometry(head_active, reset_bounding_box=False)
                    
                    # Posició: Nas (Landmark 0)
                    nose = final_landmarks[0]
                    
                    # Orientació Simplificada:
                    # Vector Espatlles (Esq a Dreta)
                    vec_x = final_landmarks[12] - final_landmarks[11] 
                    # Vector Columna (Vertical)
                    vec_y = final_landmarks[11] - final_landmarks[23] 
                    
                    # Normalitzem
                    vec_x = vec_x / (np.linalg.norm(vec_x) + 1e-6)
                    vec_y = vec_y / (np.linalg.norm(vec_y) + 1e-6)
                    
                    # Vector Endavant (Z) = Producte Vectorial (X * Y)
                    vec_z = np.cross(vec_x, vec_y)
                    
                    # Creem la matriu de rotació [X, Y, Z] directament
                    # (Assumint que el teu OBJ mira cap a +Z)
                    R = np.column_stack((vec_x, vec_y, vec_z))

                    # Apliquem al mesh
                    head_active = head_base.clone()
                    head_active.rotate(R, center=(0,0,0))
                    head_active.translate(nose)
                    
                    vis.add_geometry(head_active, reset_bounding_box=False)

            vis.poll_events()
            vis.update_renderer()
            
            # Captura
            img = (np.asarray(vis.capture_screen_float_buffer(True)) * 255).astype(np.uint8)[:, :, [2, 1, 0]]
            with self.frame_lock: latest_virtual_frame = img
            time.sleep(1/30)

    def stop(self):
        self.running = False
        
# ==========================================
# 4. STREAMING FLASK Y RUTAS
# ==========================================

def generate_frames_real(camera_stream_obj):
    """Generador para los streams reales (lee del gestor de cámara)."""
    while True:
        frame = camera_stream_obj.get_frame() 
        if frame is None:
            time.sleep(0.01)
            continue
        
        # Codificar a JPEG para el streaming MJPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(1/30)

def generate_frames_virtual():
    """Generador para el stream virtual (Lee el frame 3D del hilo)."""
    global latest_virtual_frame
    
    # Frame de error inicial
    error_frame = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    cv2.putText(error_frame, "3D PROCESS STARTING...", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    ret_err, buffer_err = cv2.imencode('.jpg', error_frame)
    error_frame_bytes = buffer_err.tobytes()

    while True:
        frame_to_stream = None
        with frame_lock:
            if latest_virtual_frame is not None:
                frame_to_stream = latest_virtual_frame
        
        if frame_to_stream is not None:
            # Codificar el frame 3D más reciente a JPEG
            ret, buffer = cv2.imencode('.jpg', frame_to_stream)
            frame_bytes = buffer.tobytes()
        else:
            frame_bytes = error_frame_bytes
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(1/30)

# ----------------- RUTAS DE VIDEO -----------------

@app.route('/video_feed_real1')
def video_feed_real1():
    global camera_front
    return Response(generate_frames_real(camera_front),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    
@app.route('/video_feed_real2')
def video_feed_real2():
    global camera_side
    return Response(generate_frames_real(camera_side),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_virtual')
def video_feed_virtual():
    return Response(generate_frames_virtual(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================
# 5. INICIO DEL SERVIDOR
# ==========================================

if __name__ == '__main__':
    # 1. Iniciar los hilos de captura de cámara
    print("Starting Flask server and 3D processing thread...")
    camera_front = CameraStream(0)
    camera_side = CameraStream(1)
    
    # Lógica de prueba alternativa para cámaras (útil si los índices no son 0 y 1)
    if not camera_side.cap.isOpened():
        print("Cámara 1 no se pudo abrir. Probando índice 2...")
        camera_side.stop()
        camera_side = CameraStream(2) 
    
    if not camera_front.cap.isOpened() and camera_side.cap.isOpened():
        print("Cámara 0 no se pudo abrir. Probando índice 1 para la frontal...")
        camera_front.stop()
        camera_front = CameraStream(1)

    camera_front.start()
    camera_side.start()
    
    # 2. Iniciar el hilo de procesamiento 3D
    processor_thread = ReconstructionProcessor(camera_front, camera_side, frame_lock)
    processor_thread.start()
    
    # 3. Iniciar el servidor Flask
    try:
        print("Servidor Flask iniciado y procesamiento 3D en segundo plano.")
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        # 4. Limpieza y cierre de recursos
        print("Cerrando servidor y limpiando recursos...")
        if processor_thread and processor_thread.is_alive():
            processor_thread.stop()
            processor_thread.join()
        if camera_front and camera_front.is_alive():
            camera_front.stop()
            camera_front.join()
        if camera_side and camera_side.is_alive():
            camera_side.stop()
            camera_side.join()
        # LA LÍNEA cv2.destroyAllWindows() SE HA ELIMINADO AQUÍ PARA EVITAR ERRORES.