# app.py
from flask import Flask, render_template, jsonify, send_file, request
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64
from threading import Lock
import os
from pathlib import Path

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


modelo = joblib.load('modelo_posturas.pkl')
encoder = joblib.load('encoder.pkl')


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)


thread_lock = Lock()
thread = None
current_reference_landmarks = None  # Landmarks de la imagen de referencia (single)
current_reference_sequence = []    # Lista de landmarks por frame del video de referencia
reference_fps = 1                  # FPS de la secuencia de referencia (frames por segundo)
current_reference_index = 0        # Índice de frame de referencia sincronizado desde cliente
user_landmarks_buffer = []         # Buffer corto de landmarks del usuario para suavizar ruido
current_tolerance = 1.0            # Factor de tolerancia: >1 más permisivo, <1 más estricto

# Nota: Ahora se asume que el modelo devuelve directamente la etiqueta
# correspondiente a la condición médica (p. ej. 'lumbalgia mecanica inespecifica',
# 'hernia de disco lumbar', 'espondilolisis', 'escoliosis lumbar', etc.).


def evaluate_posture(landmarks, posture_label, reference_landmarks=None, tolerance_scale=1.0):
    """Evaluación mejorada que devuelve 'Bien' o 'Mal' y una breve sugerencia.
    
    Utiliza umbrales CALIBRADOS basados en análisis de videos reales.
    Si se proporciona reference_landmarks, se compara con precisión basada en distancias de landmarks.
    Si no, usa heurísticas especializadas por enfermedad.
    """
    if not landmarks or not posture_label:
        return 'Sin evaluación', 'No hay datos suficientes', {'avg_distance': None, 'max_distance': None}

    # Convertir la lista plana a pares (x,y,z) por punto
    pts = [(landmarks[i], landmarks[i+1], landmarks[i+2]) for i in range(0, len(landmarks), 3)]

    # Índices comunes en MediaPipe Pose
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    NOSE = 0
    NECK = 1

    feedback = 'Bien'
    reason = 'Postura dentro de parámetros esperados'

    try:
        ls = pts[LEFT_SHOULDER]
        rs = pts[RIGHT_SHOULDER]
        lh = pts[LEFT_HIP]
        rh = pts[RIGHT_HIP]
        nose = pts[NOSE]
        
        # Si hay landmarks de referencia, calcular similitud con umbrales CALIBRADOS
        avg_distance = None
        max_distance = None
        if reference_landmarks and len(reference_landmarks) == len(landmarks):
            ref_pts = [(reference_landmarks[i], reference_landmarks[i+1], reference_landmarks[i+2]) 
                       for i in range(0, len(reference_landmarks), 3)]
            
            # Calcular distancia euclidiana promedio entre landmarks (incluyendo todas las articulaciones)
            distances = []
            for i in range(min(len(pts), len(ref_pts))):
                p = pts[i]
                ref_p = ref_pts[i]
                dist = np.sqrt((p[0] - ref_p[0])**2 + (p[1] - ref_p[1])**2 + (p[2] - ref_p[2])**2)
                distances.append(dist)
            
            avg_distance = np.mean(distances)
            max_distance = np.max(distances)
            
            # Ajustar umbrales según factor de tolerancia
            t = float(tolerance_scale)
            # Umbrales CALIBRADOS basados en análisis de frames del mismo video (promedio 0.0148)
            # Multiplicados por factores para permitir variación natural del usuario
            if avg_distance < 0.05 * t and max_distance < 0.12 * t:
                feedback = 'Bien'
                similarity_pct = max(0, 100 - int(avg_distance * 1000))
                reason = f'✓ Excelente coincidencia con la referencia ({similarity_pct}%)'
            elif avg_distance < 0.09 * t and max_distance < 0.18 * t:
                feedback = 'Bien'
                similarity_pct = max(0, 100 - int(avg_distance * 1000))
                reason = f'✓ Buena postura, muy cercana a la referencia ({similarity_pct}%)'
            elif avg_distance < 0.15 * t and max_distance < 0.30 * t:
                feedback = 'Mal'
                reason = f'⚠ La postura se desvía. Ajusta tu alineación.'
            else:
                feedback = 'Mal'
                reason = f'✗ La postura es muy diferente. Intenta nuevamente.'
        else:
            # Heurísticas especializadas si no hay referencia
            shoulder_avg_y = (ls[1] + rs[1]) / 2
            hip_avg_y = (lh[1] + rh[1]) / 2
            shoulder_avg_x = (ls[0] + rs[0]) / 2
            hip_avg_x = (lh[0] + rh[0]) / 2
            
            trunk_tilt_y = abs(hip_avg_y - shoulder_avg_y)
            trunk_tilt_x = abs(hip_avg_x - shoulder_avg_x)
            
            shoulder_diff = abs(ls[1] - rs[1])
            hip_diff = abs(lh[1] - rh[1])

            if posture_label == 'espondilolisis':
                # Requiere curvatura visible
                if trunk_tilt_y > 0.03 or trunk_tilt_x > 0.05:
                    feedback = 'Bien'
                    reason = '✓ Curvatura de espondilolisis detectada correctamente'
                else:
                    feedback = 'Mal'
                    reason = '✗ No hay suficiente curvatura. Aumenta la flexión de la columna.'
            
            elif posture_label == 'lumbalgia mecánica inespecífica':
                # Requiere alineación vertical
                if abs(trunk_tilt_y) < 0.05 and abs(trunk_tilt_x) < 0.04:
                    feedback = 'Bien'
                    reason = '✓ Alineación óptima para lumbalgia mecánica'
                else:
                    feedback = 'Mal'
                    reason = '✗ Desalineación. Alinea hombros directamente sobre caderas.'
            
            elif posture_label == 'escoliosis lumbar':
                # Requiere simetría
                if shoulder_diff < 0.02 and hip_diff < 0.02:
                    feedback = 'Bien'
                    reason = '✓ Postura simétrica, buena para escoliosis'
                else:
                    feedback = 'Mal'
                    reason = f'✗ Inclinación lateral detectada. Nivela hombros y pelvis.'
            
            elif posture_label == 'hernia de disco lumbar':
                # Requiere soporte neutral
                if trunk_tilt_y < 0.08 and abs(trunk_tilt_x) < 0.06:
                    feedback = 'Bien'
                    reason = '✓ Posición neutra segura para hernia de disco'
                else:
                    feedback = 'Mal'
                    reason = '✗ Posición de riesgo. Mantén la espalda neutral y apoyada.'
    
    except Exception as e:
        return 'Sin evaluación', f'Error al calcular landmarks: {str(e)}', {'avg_distance': None, 'max_distance': None}

    return feedback, reason, {'avg_distance': float(avg_distance) if avg_distance is not None else None,
                              'max_distance': float(max_distance) if max_distance is not None else None}

def classify_posture(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        # Extraer landmarks en forma plana [x,y,z,...]
        landmarks = [coord for landmark in results.pose_landmarks.landmark
                     for coord in [landmark.x, landmark.y, landmark.z]]

        # Predecir postura
        try:
            posture = encoder.inverse_transform(modelo.predict([landmarks]))[0]
        except Exception:
            posture = None

        # Dibujar landmarks en el frame
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return frame, posture, landmarks

    return frame, None, None

def video_stream():
    global current_reference_landmarks
    global current_reference_sequence, current_reference_index, user_landmarks_buffer
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame, posture, landmarks = classify_posture(frame)

        # Tratar la etiqueta del modelo como la condición médica
        condition = posture if posture else 'No detectada'

        # Suavizar landmarks del usuario (promedio último par de frames) para reducir ruido
        if landmarks:
            user_landmarks_buffer.append(landmarks)
            # mantener buffer corto
            if len(user_landmarks_buffer) > 3:
                user_landmarks_buffer.pop(0)
            # promediar elemento a elemento
            avg_landmarks = None
            try:
                arr = np.array(user_landmarks_buffer)
                avg_landmarks = list(np.mean(arr, axis=0))
            except Exception:
                avg_landmarks = landmarks
        else:
            avg_landmarks = None

        # Seleccionar landmarks de referencia sincronizados si existe una secuencia
        ref_landmarks_to_use = None
        if current_reference_sequence:
            idx = min(max(0, current_reference_index), len(current_reference_sequence)-1)
            ref_landmarks_to_use = current_reference_sequence[idx]
        elif current_reference_landmarks:
            ref_landmarks_to_use = current_reference_landmarks

        # Evaluar feedback (Bien/Mal + razón) usando landmarks de referencia sincronizados si están disponibles
        feedback_label, feedback_reason, metrics = evaluate_posture(avg_landmarks, posture, ref_landmarks_to_use, tolerance_scale=current_tolerance)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        with thread_lock:
            socketio.emit('video_feed', {
                'image': f"data:image/jpeg;base64,{frame_base64}",
                'posture': posture or "No detectado",
                'condition': condition,
                'feedback': feedback_label,
                'reason': feedback_reason,
                'metrics': metrics
            })
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/datasets')
def get_datasets():
    """Retorna la lista de datasets disponibles (carpetas en 'dataset/')"""
    dataset_dir = Path('dataset')
    if not dataset_dir.exists():
        return jsonify([])
    
    datasets = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    return jsonify(sorted(datasets))

@app.route('/api/images/<dataset>')
def get_images(dataset):
    """Retorna la lista de videos disponibles en un dataset"""
    dataset_path = Path('dataset') / dataset
    if not dataset_path.exists():
        return jsonify([])
    
    # Extensiones de video soportadas
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    videos = [f.name for f in dataset_path.iterdir() 
              if f.is_file() and f.suffix.lower() in video_extensions]
    return jsonify(sorted(videos))

@app.route('/api/video/<dataset>/<video>')
def get_video(dataset, video):
    """Sirve un video específico del dataset"""
    video_path = Path('dataset') / dataset / video
    
    # Validar que el archivo existe y tiene extensión válida
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    if not video_path.exists() or video_path.suffix.lower() not in video_extensions:
        return jsonify({'error': 'Video no encontrado'}), 404
    
    return send_file(video_path, mimetype='video/mp4')

@socketio.on('set_reference_video')
def handle_set_reference_video(data):
    """Recibe la ruta del video de referencia, extrae una secuencia de landmarks muestreada
    y la guarda en `current_reference_sequence`. Se expone `reference_fps` para sincronización.
    Parámetros esperados en `data`: 'video_path' y opcional 'target_fps' (por defecto 2).
    """
    global current_reference_landmarks, current_reference_sequence, reference_fps, current_reference_index

    video_path = data.get('video_path')
    if not video_path:
        return

    full_path = Path(video_path)
    if not full_path.exists():
        socketio.emit('reference_video_set', {'success': False, 'message': 'Video no encontrado'})
        return

    try:
        # Abrir el video
        cap = cv2.VideoCapture(str(full_path))
        if not cap.isOpened():
            socketio.emit('reference_video_set', {'success': False, 'message': 'No se pudo abrir el video'})
            return

        # Configurar muestreo
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_fps = float(data.get('target_fps', 2))
        reference_fps = max(1, int(target_fps))
        step = max(1, int(round(video_fps / target_fps)))

        seq = []
        frame_idx = 0
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Para evitar procesar videos larguísimos, limitamos a 300 muestras
        max_samples = 300
        samples = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                # Procesar con MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                if results.pose_landmarks:
                    landmarks = [coord for landmark in results.pose_landmarks.landmark
                                 for coord in [landmark.x, landmark.y, landmark.z]]
                    seq.append(landmarks)
                    samples += 1
                    if samples >= max_samples:
                        break

            frame_idx += 1

        cap.release()

        if not seq:
            socketio.emit('reference_video_set', {'success': False, 'message': 'No se detectaron landmarks en el video'})
            return

        # Guardar secuencia y primera referencia
        current_reference_sequence = seq
        current_reference_landmarks = seq[0]
        current_reference_index = 0

        socketio.emit('reference_video_set', {
            'success': True,
            'frames': len(seq),
            'ref_fps': reference_fps,
            'message': f'Se extrajeron {len(seq)} frames de referencia (muestreo {reference_fps} fps)'
        })

    except Exception as e:
        socketio.emit('reference_video_set', {'success': False, 'message': str(e)})


@socketio.on('sync_reference_time')
def handle_sync_reference_time(data):
    """Recibe {'current_time': seconds} del cliente para sincronizar el índice de referencia."""
    global current_reference_index, reference_fps
    try:
        t = float(data.get('current_time', 0))
        idx = int(round(t * reference_fps))
        current_reference_index = max(0, idx)
    except Exception:
        pass


@socketio.on('set_tolerance')
def handle_set_tolerance(data):
    """Recibe {'tolerance': float} para ajustar la tolerancia de comparación en tiempo real."""
    global current_tolerance
    try:
        t = float(data.get('tolerance', 1.0))
        # limitar rango razonable
        if t <= 0:
            return
        current_tolerance = max(0.3, min(3.0, t))
        socketio.emit('tolerance_updated', {'tolerance': current_tolerance})
    except Exception:
        pass

@socketio.on('connect')
def handle_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(video_stream)


@app.route('/api/evaluate_frame', methods=['POST'])
def evaluate_frame():
    """Endpoint para evaluar una imagen enviada por la app móvil.
    Espera JSON: {'image': 'data:image/jpeg;base64,...'}
    Devuelve: posture, feedback, reason, metrics
    """
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({'success': False, 'message': 'No JSON payload'}), 400

        img_b64 = payload.get('image')
        if not img_b64:
            return jsonify({'success': False, 'message': 'No image provided'}), 400

        # Soporta data URI o solo base64
        if ',' in img_b64:
            img_b64 = img_b64.split(',', 1)[1]

        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'success': False, 'message': 'Invalid image data'}), 400

        # Usar las funciones existentes para clasificar y evaluar
        frame_proc, posture, landmarks = classify_posture(frame)

        # Seleccionar referencia sincronizada si existe
        ref_landmarks_to_use = None
        if current_reference_sequence:
            idx = min(max(0, current_reference_index), len(current_reference_sequence)-1)
            ref_landmarks_to_use = current_reference_sequence[idx]
        elif current_reference_landmarks:
            ref_landmarks_to_use = current_reference_landmarks

        feedback_label, feedback_reason, metrics = evaluate_posture(landmarks, posture, ref_landmarks_to_use, tolerance_scale=current_tolerance)

        return jsonify({'success': True,
                        'posture': posture,
                        'feedback': feedback_label,
                        'reason': feedback_reason,
                        'metrics': metrics})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/sync_reference_time', methods=['POST'])
def sync_reference_time_http():
    """Endpoint HTTP para sincronizar tiempo del video (para app móvil).
    Espera JSON: {'current_time': seconds}
    Actualiza current_reference_index basado en el tiempo.
    """
    global current_reference_index, reference_fps
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({'success': False}), 400
        
        t = float(payload.get('current_time', 0))
        idx = int(round(t * reference_fps))
        current_reference_index = max(0, idx)
        
        return jsonify({'success': True, 'index': current_reference_index})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


if __name__ == '__main__':
    # Usar socketio.run para mantener soporte SocketIO + Flask routes HTTP
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)