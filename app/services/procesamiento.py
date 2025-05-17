import json
import cv2
import mediapipe as mp
import time
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
"""from app.services.LecturaJson import GuardadCSv
from app.services.LecturaDataSet import LecturaDataSet
from app.services.PreProcesamiento import PreProcesamiento"""
from LecturaJson import GuardadCSv
from LecturaDataSet import LecturaDataSet
from PreProcesamiento import PreProcesamiento
from concurrent.futures import ProcessPoolExecutor


model_path = "resources/letras_LSTM.h5"
rutaDataSet = "resources/DataSet/Entrenamiento"
scaler_path = "resources/scaler.pkl"
label_encoder_path = "resources/label_encoder.pkl"
rutaCSV = "resources/corpues2.csv"
clases = ["A","B","C","D","E","F","G","H","I", "J", "K", "L", "M", "N", "Ñ", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
frames_data = []
gaurdar = GuardadCSv()
preProceso = PreProcesamiento()

model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)
"""model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)"""

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
FACE_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION

start_time = time.time()
frame_counter = 0
last_capture_time = start_time


def preparar_datos(json_input):
    """Convierte los datos de entrada al formato adecuado para el modelo"""
    timesteps = 5
    try:
        # 1. Extraer características (ajusta según tu implementación)
        x = np.array(json_input)

        # 2. Verificar dimensiones
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # 3. Escalar características
        x_scaled = scaler.transform(x)

        # 4. Redimensionar para LSTM (timesteps, features)
        # Si tienes menos muestras que timesteps, necesitas padding o ajuste
        if x_scaled.shape[0] < timesteps:
            # Padding con ceros (alternativa: replicar los datos)
            padding = np.zeros((timesteps - x_scaled.shape[0], x_scaled.shape[1]))
            x_scaled = np.vstack([x_scaled, padding])

        # Crear secuencia temporal
        x_ready = x_scaled.reshape(1, timesteps, -1)

        return x_ready

    except Exception as e:
        print(f"Error en preparación de datos: {str(e)}")
        raise
def clasificar(json_input):
    """Realiza la clasificación de la seña"""
    umbral_confianza = 0.20
    try:
        # 1. Preparar datos
        x_ready = preparar_datos(json_input)
        print(f"Forma de los datos preparados: {x_ready.shape}")

        # 2. Realizar predicción
        predicciones = model.predict(x_ready, verbose=0)

        # 3. Procesar resultados
        clase_idx = np.argmax(predicciones)
        confianza = float(np.max(predicciones))

        # 4. Validar resultados
        if confianza < umbral_confianza:
            return "not", confianza

        if clase_idx >= len(label_encoder.classes_):
            return "Error: Clase desconocida", confianza

        # Decodificar la etiqueta
        clase_nombre = label_encoder.inverse_transform([clase_idx])[0]

        return clase_nombre, confianza

    except Exception as e:
        print(f"Error en clasificación: {str(e)}")
        return "Error en procesamiento", 0.0


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


def mostrarCamara(prediccion, confianza, frame):
    cv2.putText(frame,f"{prediccion} ({confianza:.0%})",(10, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2,)
    cv2.imshow("Prediccion", frame)

def video(i,last_capture_time, cap, nomClase, puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq):
     with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        
        width = int(cap.get(3))
        height = int(cap.get(4))
        center = (width // 2, height // 2)
        fps = int(cap.get(5))
        conta = 0
        datosscvlist = []
        
        while True:
           
            ret, frame = cap.read()
            if not ret:
                break
            current_time = time.time()
            elapsed_time = current_time - last_capture_time
            if elapsed_time >= 0.06:
                conta = conta + 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rotation_matrix = cv2.getRotationMatrix2D(center, i, 1.0)
                rotated_frame = cv2.warpAffine(frame_rgb, rotation_matrix, (width, height))
                results = holistic.process(rotated_frame)
                tempPuntosDer = puntosDer
                tempPuntosIzq = puntosIzq
                puntosDer = []
                puntosIzq = []
                frame_data = {"id": len(frames_data) + 1, "datos_brazos": {}}
                frame = cv2.flip(frame, 1)
                last_capture_time = current_time
                frame_data, puntosDer, puntosIzq = preProceso.pendiente(results, frame_data, puntosDer, puntosIzq)
                #========================PRE-PROCESAMIENTO=============================
                preProceso.setJson_P(frame_data)
                #======================================================================
                frame_data = preProceso.variacion(frame_data, puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq)
                #datosList.append(frame_data)
                gaurdar.guardadDatos(frame_data, nomClase, rutaCSV)
        print("---------------------------------------------->>>>>>>"+str(conta))  

        """for i in datosList:
            for j in gaurdar.extraccion(i):
                datosscvlist.append(j)
        if bool(datosscvlist):
            gaurdar.guardadDatosVideo(datosscvlist, nomClase, rutaCSV)"""
                
     cap.release()
     cv2.destroyAllWindows()

def camara(opc, last_capture_time, cap, nomClase, puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq):
     with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_time = time.time()
            elapsed_time = current_time - last_capture_time
            if elapsed_time >= 0.2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)


                tempPuntosDer = puntosDer
                tempPuntosIzq = puntosIzq
                puntosDer = []
                puntosIzq = []
                frame_data = {"id": len(frames_data) + 1, "datos_brazos": {}}
                frame = cv2.flip(frame, 1)
                last_capture_time = current_time
                """ print("==================================PUNTOS==============================================")
                for i in enumerate(results.right_hand_landmarks.landmark):
                    print("================" + str(i[0]))
                    print(i[1])"""
        
                frame_data, puntosDer, puntosIzq = preProceso.pendiente(results, frame_data, puntosDer, puntosIzq)
                #========================PRE-PROCESAMIENTO=============================
                print(preProceso.movimiento(frame_data))
                preProceso.setJson_P(frame_data)
                #======================================================================
                frame_data = preProceso.variacion(frame_data, puntosDer, puntosIzq, tempPuntosDer, tempPuntosIzq)
                prediccion, confianza = clasificacion(frame_data)
                #========================IMPRESIONES EN CONSOLA=========================
                """json_formateado = json.dumps(frame_data, indent=4, ensure_ascii=False)
                print(json_formateado)"""
                #print(f"Predicción: {prediccion} (Confianza: {confianza:.2f})")
                #========================================================================
                if opc == 1:
                    mostrarCamara(prediccion, confianza, frame)
                elif opc == 2:
                    gaurdar.guardadDatos(frame_data, nomClase, rutaCSV)
                elif opc == 3:
                    key = gaurdar.validacion(prediccion, frame_data, rutaCSV, clases)
                    if  key == 27:
                       if ( os.path.exists(rutaCSV)and input("¿Reentrenar el modelo? (s/n): ").strip().lower() == "s"): gaurdar.reentrenar_modelo(rutaCSV, model_path, model, scaler, clases)
                       break
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
     cap.release()
     cv2.destroyAllWindows()


def webCam(ima, datosList, results_Tem):
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2) as holistic:
        jsonTemp = {}
        image = ima
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = holistic.process(image_rgb)
        frame_data = {"id": len(frames_data) + 1, "datos_brazos": {}}
        frame_data = preProceso.pendiente(results, frame_data)
        frame_data = preProceso.angulos(frame_data)
        #frame_data = preProceso.velocidad_magnitud(frame_data, results, results_Tem)
        frame_data = preProceso.movimiento_angulos(frame_data, jsonTemp)
        
        datosList.append(frame_data)
        results_Tem = results
        
        

        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return datosList, results_Tem

def videoMov(last_capture_time, captura, nomClase, angulo):
    print("=======================================INICIO============================")
    results_Tem = {}
    cap = cv2.VideoCapture(captura)
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:    
        datosList = []
        jsonTemp = {}
        datosscvlist = []
        conta = 0
        pendientes =[]
        while True:
            frame_data = {}
            results = {}
            ret, frame = cap.read()
            if not ret:
                break
            current_time = time.time()
            elapsed_time = current_time - last_capture_time

            if elapsed_time >= 0.035:
                conta += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #ROTACION DE LA IMAGEN
                (h,w) = frame_rgb.shape[:2]
                centro = (w//2, h//2)
                matriz_rotate = cv2.getRotationMatrix2D(centro, angulo, 1.0)
                imagen_rotate = cv2.warpAffine(frame_rgb, matriz_rotate,(w, h))
                #proceso
                results = holistic.process(imagen_rotate)
                frame_data = {"id": len(frames_data) + 1, "datos_brazos": {}}
                frame = cv2.flip(frame, 1)
                last_capture_time = current_time
                frame_data= preProceso.pendiente(results, frame_data)
                #========================PRE-PROCESAMIENTO=============================
                #======================================================================
                frame_data = preProceso.angulos(frame_data)
                #frame_data = preProceso.velocidad_magnitud(frame_data, results, results_Tem)
                frame_data = preProceso.movimiento_angulos(frame_data, jsonTemp)
                
                datosList.append(frame_data)
                #datosList.append(frame_data)
                
                results_Tem = results
                #pendientes.extend(gaurdar.getPendientes(frame_data))
                #jsonTemp = frame_data
                """json_formateado = json.dumps(frame_data, indent=4, ensure_ascii=False)
                print(json_formateado)"""
        #=======================Serie de Fourier=======================
        """newData = preProceso.getDatosImg(datosList)
        newData.extend(pendientes)
        gaurdar.guardadDatosVideo(newData, nomClase, rutaCSV)"""

        for i in datosList:
            
            for j in gaurdar.extraccion(i):
                datosscvlist.append(j)
        newData = preProceso.getDatosImg(datosList)
        """print("========================================================")
        print("========================================================")
        print("========================================================")
        print("========================================================")
        print(len(newData))"""
        
        datosscvlist.extend(newData)
        if bool(datosscvlist):
            gaurdar.guardadDatosVideo(datosscvlist, nomClase, rutaCSV)
        cap.release()
        
    cv2.destroyAllWindows()

def imagen(ruta, nomClase):
    results_Tem = {}
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2) as holistic:
        jsonTemp = {}
        image = cv2.imread(ruta)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        (h, w) = image_rgb.shape[:2]
        centro = (w // 2, h // 2)

        for i in range(-17, 16, 2):
            print("Angulo: " + str(i))
            matriz_rotacion = cv2.getRotationMatrix2D(centro, i, 1.0)
            imagen_rotada = cv2.warpAffine(image_rgb, matriz_rotacion, (w, h))
            results = holistic.process(imagen_rotada)
            frame_data = {"id": len(frames_data) + 1, "datos_brazos": {}}
            frame_data = preProceso.pendiente(results, frame_data)
            frame_data = preProceso.angulos(frame_data)
            frame_data = preProceso.velocidad_magnitud(frame_data, results, results_Tem)
            frame_data = preProceso.movimiento_angulos(frame_data, jsonTemp)
            newJson = preProceso.getDatosImg(frame_data)
            json_formateado = json.dumps(newJson, indent=4, ensure_ascii=False) 
            print(json_formateado)
            #gaurdar.guardadDatos(frame_data, nomClase, rutaCSV)

        cv2.waitKey(0)
    cv2.destroyAllWindows()
def llamar(img):
    results_Tem = {}
    datosList = []
    array =[]
    datosscvlist = []
    for i in img:
        datosList, results_Tem= webCam(i, datosList, results_Tem)
    """newData = preProceso.getDatosImg(datosList)
    newData.extend(pendientes)"""
    
    for i in datosList:
        for j in gaurdar.extraccion(i):
            datosscvlist.append(j)
        array.append(datosscvlist)
        datosscvlist = []
    prediccion, confianza = clasificacionM(array)
    print(prediccion)
    print(confianza)
    return prediccion

def cargarVideos():
    leerVideo = LecturaDataSet()
    tareas = []
    with ProcessPoolExecutor() as executor:
        for ruta in leerVideo.extraccionVideo(rutaDataSet):
            for i in range(-15, 15, 2):
                tarea = executor.submit(videoMov, last_capture_time, ruta[0], ruta[1], i)
                tareas.append(tarea)
        resultados = [t.result() for t in tareas]

def cargarVideosP():
    leerVideo = LecturaDataSet()
    tareas = []
    with ProcessPoolExecutor() as executor:
        for ruta in leerVideo.extraccionVideo(rutaDataSet):
            for i in range(-15, 15, 2):
                videoMov(last_capture_time, ruta[0], ruta[1], i)


def cargarImagenes():
     lecturaImg = LecturaDataSet()
     for ruta in lecturaImg.extraccionImagenes(rutaDataSet):
        imagen(ruta[0],ruta[1])

if __name__ == "__main__":
    cargarVideos()
