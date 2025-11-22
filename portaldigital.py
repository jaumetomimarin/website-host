import cv2
import numpy as np
import gradio as gr
import math
import time
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Inicializar las cámaras
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

# Verificar que ambas cámaras estén funcionando
if not cap1.isOpened():
    print("Error: No se pudo abrir la cámara 1")
if not cap2.isOpened():
    print("Error: No se pudo abrir la cámara 2")

backSub1 = cv2.createBackgroundSubtractorMOG2(200, 16)
backSub2 = cv2.createBackgroundSubtractorMOG2(200, 16)

# Inicializar Pygame para visualización 3D
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption("Mundo al Revés 3D")

gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)
glRotatef(25, 1, 0, 0)

# Variables para almacenar los objetos 3D detectados
detected_objects_3d = []

class Object3D:
    def __init__(self, x, y, z, size):
        self.x = x
        self.y = y
        self.z = z
        self.size = size
        self.color = (
            np.random.uniform(0.2, 1.0),
            np.random.uniform(0.2, 1.0),
            np.random.uniform(0.2, 1.0)
        )
        self.particles = []
        self.last_update = time.time()
        
    def update_particles(self):
        current_time = time.time()
        if current_time - self.last_update > 0.05:
            self.last_update = current_time
            # Agregar nuevas partículas
            for _ in range(2):
                angle = np.random.uniform(0, 2 * math.pi)
                radius = self.size * 1.5
                speed = np.random.uniform(0.01, 0.05)
                life = np.random.uniform(1.0, 3.0)
                self.particles.append({
                    'angle': angle,
                    'radius': radius,
                    'speed': speed,
                    'life': life,
                    'max_life': life
                })
            
            # Actualizar y eliminar partículas viejas
            self.particles = [p for p in self.particles if p['life'] > 0]
            for p in self.particles:
                p['life'] -= 0.1
                p['angle'] += p['speed']

def draw_cube(x, y, z, size, color):
    vertices = [
        [x-size, y-size, z-size], [x+size, y-size, z-size], [x+size, y+size, z-size], [x-size, y+size, z-size],
        [x-size, y-size, z+size], [x+size, y-size, z+size], [x+size, y+size, z+size], [x-size, y+size, z+size]
    ]
    
    edges = [
        [0,1], [1,2], [2,3], [3,0],
        [4,5], [5,6], [6,7], [7,4],
        [0,4], [1,5], [2,6], [3,7]
    ]
    
    glBegin(GL_LINES)
    glColor3f(color[0], color[1], color[2])
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_particles(obj):
    glBegin(GL_POINTS)
    for particle in obj.particles:
        if particle['life'] > 0:
            life_ratio = particle['life'] / particle['max_life']
            px = obj.x + math.cos(particle['angle']) * particle['radius'] * life_ratio
            py = obj.y + math.sin(particle['angle']) * particle['radius'] * life_ratio
            pz = obj.z + math.sin(particle['angle'] * 2) * particle['radius'] * 0.5 * life_ratio
            
            # Color que cambia con el tiempo
            r = obj.color[0] * life_ratio
            g = obj.color[1] * life_ratio
            b = obj.color[2] * life_ratio
            
            glColor3f(r, g, b)
            glVertex3f(px, py, pz)
    glEnd()

def render_3d_world():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Rotación automática de la escena
    glRotatef(0.5, 0, 1, 0)
    
    # Dibujar ejes coordenados
    glBegin(GL_LINES)
    # Eje X (rojo)
    glColor3f(1, 0, 0)
    glVertex3f(-2, 0, 0)
    glVertex3f(2, 0, 0)
    # Eje Y (verde)
    glColor3f(0, 1, 0)
    glVertex3f(0, -2, 0)
    glVertex3f(0, 2, 0)
    # Eje Z (azul)
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, -2)
    glVertex3f(0, 0, 2)
    glEnd()
    
    # Dibujar objetos 3D detectados
    for obj in detected_objects_3d:
        obj.update_particles()
        draw_cube(obj.x, obj.y, obj.z, obj.size, obj.color)
        draw_particles(obj)
    
    pygame.display.flip()

def process_dual_frames(frame1, frame2):
    global detected_objects_3d
    
    # Procesar primera cámara
    fg_mask1 = backSub1.apply(frame1, learningRate=0.7)
    retval1, mask_thresh1 = cv2.threshold(fg_mask1, 120, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_eroded1 = cv2.morphologyEx(mask_thresh1, cv2.MORPH_OPEN, kernel)
    
    # Procesar segunda cámara
    fg_mask2 = backSub2.apply(frame2, learningRate=0.7)
    retval2, mask_thresh2 = cv2.threshold(fg_mask2, 120, 255, cv2.THRESH_BINARY)
    mask_eroded2 = cv2.morphologyEx(mask_thresh2, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos en ambas cámaras
    contours1, _ = cv2.findContours(mask_eroded1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask_eroded2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos pequeños
    min_contour_area = 500
    large_contours1 = [cnt for cnt in contours1 if cv2.contourArea(cnt) > min_contour_area]
    large_contours2 = [cnt for cnt in contours2 if cv2.contourArea(cnt) > min_contour_area]
    
    # Dibujar contornos en los frames reales
    frame_real1 = frame1.copy()
    frame_real2 = frame2.copy()
    cv2.drawContours(frame_real1, large_contours1, -1, (0, 255, 0), 2)
    cv2.drawContours(frame_real2, large_contours2, -1, (0, 255, 0), 2)
    
    # Crear representación del Mundo al Revés 2D
    frame_upside_down1 = create_upside_down_world(frame1, large_contours1)
    frame_upside_down2 = create_upside_down_world(frame2, large_contours2)
    
    # Actualizar objetos 3D basados en la detección estereoscópica
    update_3d_objects(large_contours1, large_contours2, frame1.shape)
    
    return frame_real1, frame_real2, frame_upside_down1, frame_upside_down2

def update_3d_objects(contours1, contours2, frame_shape):
    global detected_objects_3d
    
    height, width = frame_shape[:2]
    
    # Limitar el número de objetos para mejor rendimiento
    max_objects = 8
    contours1 = contours1[:max_objects]
    contours2 = contours2[:max_objects]
    
    # Actualizar objetos existentes o crear nuevos
    new_objects = []
    
    for i, (cnt1, cnt2) in enumerate(zip(contours1, contours2)):
        if i >= len(detected_objects_3d):
            # Crear nuevo objeto 3D
            M1 = cv2.moments(cnt1)
            M2 = cv2.moments(cnt2)
            
            if M1["m00"] != 0 and M2["m00"] != 0:
                # Calcular posición 2D en ambas cámaras
                cx1 = int(M1["m10"] / M1["m00"]) / width
                cy1 = int(M1["m01"] / M1["m00"]) / height
                
                cx2 = int(M2["m10"] / M2["m00"]) / width
                cy2 = int(M2["m01"] / M2["m00"]) / height
                
                # Estimación simple de profundidad basada en la diferencia horizontal
                depth = abs(cx1 - cx2) * 2
                
                # Calcular posición 3D
                x_3d = (cx1 + cx2 - 1) * 2  # Centrado en 0, rango [-2, 2]
                y_3d = (1 - (cy1 + cy2) / 2) * 2  # Invertir Y, rango [-2, 2]
                z_3d = -depth  # Profundidad negativa (más lejos)
                
                size = math.sqrt(cv2.contourArea(cnt1)) / 100
                size = max(0.1, min(0.5, size))  # Limitar tamaño
                
                new_objects.append(Object3D(x_3d, y_3d, z_3d, size))
    
    # Actualizar la lista global de objetos
    if new_objects:
        detected_objects_3d = new_objects[:max_objects]

def create_upside_down_world(frame, contours):
    height, width = frame.shape[:2]
    
    # Invertir colores (efecto negativo)
    upside_down = 255 - frame
    
    # Aplicar distorsión ondulada
    upside_down = apply_wave_distortion(upside_down)
    
    # Añadir efecto de escáner (líneas que se mueven)
    upside_down = add_scan_lines(upside_down)
    
    # Dibujar contornos en rojo brillante para el Mundo al Revés
    cv2.drawContours(upside_down, contours, -1, (0, 0, 255), 3)
    
    # Añadir partículas alrededor de los contornos
    upside_down = add_particles(upside_down, contours)
    
    return upside_down

def apply_wave_distortion(frame):
    height, width = frame.shape[:2]
    distorted = frame.copy()
    
    # Crear un efecto de onda más simple para mejor rendimiento
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            offset_x = int(5 * math.sin(2 * math.pi * y / 60 + time.time()))
            offset_y = int(3 * math.cos(2 * math.pi * x / 80 + time.time()))
            
            new_x = (x + offset_x) % width
            new_y = (y + offset_y) % height
            
            if new_y < height and new_x < width:
                distorted[y:min(y+2, height), x:min(x+2, width)] = frame[new_y:min(new_y+2, height), new_x:min(new_x+2, width)]
    
    return distorted

def add_scan_lines(frame):
    height, width = frame.shape[:2]
    
    scan_line_pos = int((time.time() * 50) % height)
    cv2.line(frame, (0, scan_line_pos), (width, scan_line_pos), (0, 255, 255), 2)
    
    return frame

def add_particles(frame, contours):
    height, width = frame.shape[:2]
    
    for contour in contours:
        if len(contour) > 0:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                for i in range(10):
                    angle = 2 * math.pi * i / 10
                    radius = 20 + 10 * math.sin(time.time() * 3 + i)
                    px = int(cx + radius * math.cos(angle))
                    py = int(cy + radius * math.sin(angle))
                    
                    if 0 <= px < width and 0 <= py < height:
                        color = (
                            int(128 + 127 * math.sin(time.time() * 2)),
                            int(128 + 127 * math.sin(time.time() * 3 + 1)),
                            int(128 + 127 * math.sin(time.time() * 4 + 2))
                        )
                        cv2.circle(frame, (px, py), 2, color, -1)
    
    return frame

def video_stream():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return black_frame, black_frame, black_frame, black_frame
    
    # Procesar los frames de ambas cámaras
    real_world1, real_world2, upside_down1, upside_down2 = process_dual_frames(frame1, frame2)
    
    # Renderizar el mundo 3D
    render_3d_world()
    
    # Capturar el frame OpenGL
    pygame.display.flip()
    buffer = glReadPixels(0, 0, display[0], display[1], GL_RGB, GL_UNSIGNED_BYTE)
    world_3d_image = np.frombuffer(buffer, dtype=np.uint8).reshape(display[1], display[0], 3)
    world_3d_image = np.flipud(world_3d_image)  # Voltear verticalmente
    world_3d_image = cv2.cvtColor(world_3d_image, cv2.COLOR_RGB2BGR)
    
    # Convertir de BGR a RGB para Gradio
    real_world1_rgb = cv2.cvtColor(real_world1, cv2.COLOR_BGR2RGB)
    real_world2_rgb = cv2.cvtColor(real_world2, cv2.COLOR_BGR2RGB)
    upside_down1_rgb = cv2.cvtColor(upside_down1, cv2.COLOR_BGR2RGB)
    upside_down2_rgb = cv2.cvtColor(upside_down2, cv2.COLOR_BGR2RGB)
    
    return real_world1_rgb, real_world2_rgb, upside_down1_rgb, upside_down2_rgb, world_3d_image

# CSS personalizado para tema oscuro
custom_css = """
body {
    background-color: #1a1a1a;
    color: #ffffff;
}
.gr-block {
    background-color: #2d2d2d;
}
.gr-box {
    background-color: #2d2d2d;
    border: 1px solid #444;
}
"""

# Crear la interfaz de Gradio
with gr.Blocks(css=custom_css, title="Portal Digital 3D: El Mundo al Revés") as demo:
    gr.Markdown("""
    # 🌌 Portal Digital 3D: El Mundo al Revés
    
    **Sistema de Visión Estereoscópica con Proyección Tridimensional**
    
    Este portal avanzado utiliza dos cámaras para detectar movimiento en el mundo real
    y proyectarlo en tiempo real en un mundo tridimensional distorsionado.
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📷 Cámara 1 - Mundo Real")
            real_world1_output = gr.Image(label="Vista Cámara 1")
        
        with gr.Column():
            gr.Markdown("### 📷 Cámara 2 - Mundo Real")
            real_world2_output = gr.Image(label="Vista Cámara 2")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🌀 Cámara 1 - Mundo al Revés")
            upside_down1_output = gr.Image(label="Reflejo Distorsionado 1")
        
        with gr.Column():
            gr.Markdown("### 🌀 Cámara 2 - Mundo al Revés")
            upside_down2_output = gr.Image(label="Reflejo Distorsionado 2")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎮 Proyección 3D en Tiempo Real")
            world_3d_output = gr.Image(label="Mundo Tridimensional")
    
    # Actualizar en tiempo real
    demo.load(
        video_stream, 
        inputs=None, 
        outputs=[real_world1_output, real_world2_output, upside_down1_output, upside_down2_output, world_3d_output], 
        every=0.1
    )

# Si el script se ejecuta directamente
if __name__ == "__main__":
    try:
        demo.launch(share=True, inbrowser=True)
    except KeyboardInterrupt:
        print("Cerrando portal 3D...")
    finally:
        # Liberar recursos
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        pygame.quit()