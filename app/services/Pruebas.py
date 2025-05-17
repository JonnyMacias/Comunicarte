import os

def eliminar_avi_en_directorio(base_path):
    archivos_borrados = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.avi'):
                ruta_completa = os.path.join(root, file)
                try:
                    os.remove(ruta_completa)
                    archivos_borrados.append(ruta_completa)
                except Exception as e:
                    print(f"Error al borrar {ruta_completa}: {e}")

    if archivos_borrados:
        print(f"\nSe eliminaron {len(archivos_borrados)} archivos .avi:")
        for archivo in archivos_borrados:
            print(f" - {archivo}")
    else:
        print("No se encontraron archivos .avi para eliminar.")

if __name__ == "__main__":
    ruta_base = "resources/DataSet/Palabras/"

    if not os.path.isdir(ruta_base):
        print("❌ Ruta no válida. Asegúrate de escribir una carpeta existente.")
    else:
        eliminar_avi_en_directorio(ruta_base)


"""import os
import subprocess
from pathlib import Path

# Ruta base donde están las carpetas con los videos
BASE_DIR = Path("resources/DataSet/Palabras")

# Recorre todas las subcarpetas
for avi_path in BASE_DIR.rglob("*.avi"):
    mp4_path = avi_path.with_suffix(".mp4")

    if mp4_path.exists():
        print(f"Ya convertido: {mp4_path}")
        continue

    print(f"Convirtiendo: {avi_path} → {mp4_path}")

    comando = [
        "ffmpeg",
        "-i", str(avi_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        str(mp4_path)
    ]

    try:
        subprocess.run(comando, check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Error al convertir {avi_path}")
"""


"""import cv2
import numpy as np
import mediapipe as mp
import time
import tensorflow as tf
import joblib
from LecturaJson import GuardadCSv
from LecturaDataSet import LecturaDataSet
from PreProcesamiento import PreProcesamiento
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
gaurdar = GuardadCSv()
preProceso = PreProcesamiento()

# Config
SECUENCIA_FRAMES = 40
model_path = "resources/letras_LSTM.h5"

model = load_model('resources/Letras_LSTM_doble_entrada.keras')
# Variables de acumulación
datosList = []
results_Tem = None
frames_capturados = 0

cap = cv2.VideoCapture(0)
def clasificacionM(array, arrayFourier):
    try:
        # Convertir inputs a numpy
        x_seq = np.array(array, dtype=np.float32)
        x_fourier = np.array(arrayFourier, dtype=np.float32)

        # Cargar modelos y normalizadores
        scaler_seq = joblib.load('resources/scaler_seq.pkl')
        scaler_fourier = joblib.load('resources/scaler_fourier.pkl')
        label_encoder = joblib.load('resources/label_encoder.pkl')
        clases = label_encoder.classes_

        # Parámetros de forma
        expected_seq_len = scaler_seq.mean_.shape[0]  # 7470
        n_features = 90
        n_timesteps = expected_seq_len // n_features

        # --- Preprocesar secuencia ---
        if x_seq.ndim == 1:
            x_seq = x_seq.reshape(-1, n_features)
        
        # Padding o recorte
        if x_seq.shape[0] < n_timesteps:
            pad_len = n_timesteps - x_seq.shape[0]
            x_seq = np.pad(x_seq, ((0, pad_len), (0, 0)), mode='constant', constant_values=0.0)
        elif x_seq.shape[0] > n_timesteps:
            x_seq = x_seq[:n_timesteps]
        
        x_seq_flat = x_seq.reshape(1, -1)
        x_seq_scaled = scaler_seq.transform(x_seq_flat).reshape(1, n_timesteps, n_features)

        # --- Preprocesar Fourier ---
        x_fourier = x_fourier.reshape(1, -1)  # Asegurar forma (1, 320)
        x_fourier_scaled = scaler_fourier.transform(x_fourier)

        # --- Cargar modelo (si no está cargado globalmente) ---
        global model
        if 'model' not in globals():
            from tensorflow.keras.models import load_model
            model = load_model('resources/Letras_LSTM_doble_entrada.h5')

        # --- Predicción ---
        print("Forma secuencia:", x_seq_scaled.shape)
        print("Forma fourier:", x_fourier_scaled.shape)

        predicciones = model.predict([x_seq_scaled, x_fourier_scaled], verbose=0)
        clase_idx = np.argmax(predicciones)
        confianza = np.max(predicciones)

        if confianza < 0.60:
            return "not", confianza
        if clase_idx >= len(clases):
            return "Error: Clase desconocida", confianza

        return clases[clase_idx], confianza

    except Exception as e:
        print(f"Error en clasificación: {str(e)}")
        return "Error en procesamiento", 0.0


with mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=2) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el frame")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        jsonTemp = {}
        frame_data = {"id": len(datosList) + 1, "datos_brazos": {}}
        frame_data = preProceso.pendiente(results, frame_data)
        frame_data = preProceso.angulos(frame_data)
        frame_data = preProceso.movimiento_angulos(frame_data, jsonTemp)

        datosList.append(frame_data)
        results_Tem = results
        frames_capturados += 1

        # Dibujar texto del frame
        cv2.putText(frame, f'Frames capturados: {frames_capturados}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Mostrar el frame
        cv2.imshow('Captura en tiempo real', frame)

        # Cuando tengamos suficientes frames, clasificamos
        if frames_capturados >= SECUENCIA_FRAMES:
            datosscvlist = []
            for i in datosList:
                datosscvlist.extend(gaurdar.extraccion(i))
            arrayFourier = preProceso.getDatosImg(datosList)
            prediccion, confianza = clasificacionM(datosscvlist, arrayFourier)
            print(f"Predicción: {prediccion}, Confianza: {confianza:.2f}")

            # Reiniciar para la siguiente secuencia
            datosList = []
            frames_capturados = 0

        # Salir con ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()


"""

"""import csv

# Leer el archivo CSV
with open('resources/corpues2.csv', 'r') as archivo:
    lector = csv.reader(archivo)
    filas = list(lector)

# Calcular el número máximo de columnas sin contar el identificador
max_longitud = max(len(fila) - 1 for fila in filas)

# Rellenar con ceros las filas que tengan menos columnas
filas_rellenadas = []
for fila in filas:
    datos = fila[:-1]  # Todo menos el identificador
    identificador = fila[-1]  # El identificador es siempre la última columna
    datos += ['0'] * (max_longitud - len(datos))  # Rellenar con ceros
    fila_nueva = datos + [identificador]  # Reunir datos + identificador
    filas_rellenadas.append(fila_nueva)

# Guardar el archivo modificado
with open('resources/limpio.csv', 'w', newline='') as archivo:
    escritor = csv.writer(archivo)
    escritor.writerows(filas_rellenadas)
"""