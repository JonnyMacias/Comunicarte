import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

class GuardadCSv:
    def __init__(self):
        pass
    def getPendientes(self, json):
        datos_brazo = json["datos_brazos"]
        caracteristicas = []
        for mano in ["Mano Derecha", "Mano Izquierda"]:
            claves_ordenadas = sorted(datos_brazo[mano].keys())
            for clave in claves_ordenadas:
                caracteristicas.append(datos_brazo[mano][clave])
        return caracteristicas

    def extraccion(self, json_data):
        datos_brazo = json_data["datos_brazos"]
        caracteristicas = []

        caracteristicas.append(datos_brazo["Brazo Derecho"]["Brazo"])
        caracteristicas.append(datos_brazo["Brazo Derecho"]["Antebrazo"])

        caracteristicas.append(datos_brazo["Brazo Izquierdo"]["Brazo"])
        caracteristicas.append(datos_brazo["Brazo Izquierdo"]["Antebrazo"])

        for mano in ["Mano Derecha", "Mano Izquierda"]:
            claves_ordenadas = sorted(datos_brazo[mano].keys())
            for clave in claves_ordenadas:
                caracteristicas.append(datos_brazo[mano][clave])
        
        for mano in ["ang_Derecha", "ang_Izquierda"]:
            claves_ordenadas = sorted(datos_brazo[mano].keys())
            for clave in claves_ordenadas:
                caracteristicas.append(datos_brazo[mano][clave])
        
        """for mano in ["vel_Derecha", "vel_Izquierdo"]:
            claves_ordenadas = sorted(datos_brazo[mano].keys())
            for clave in claves_ordenadas:
                caracteristicas.append(datos_brazo[mano][clave])"""
        
        for mano in ["mAng_Derecha", "mAng_Izquierda"]:
            claves_ordenadas = sorted(datos_brazo[mano].keys())
            for clave in claves_ordenadas:
                caracteristicas.append(datos_brazo[mano][clave])


        return np.array(caracteristicas)
    
    def guardadDatos(self, json, clase, ruta):
        x = self.extraccion(json)
        print("............................."+str(len(x)))
        df_nuevo = pd.DataFrame([x], columns=[f"feat_{i}" for i in range(len(x))])
        df_nuevo["clase"] = clase
        if os.path.exists(ruta):
            df_nuevo.to_csv(ruta, mode="a", header=False, index=False)
        else:
            df_nuevo.to_csv(ruta, index=False)

        print(f"Datos corregidos guardados para la clase {clase}")
    
    def guardadDatosVideo(self, x, clase, ruta):
        df_nuevo = pd.DataFrame([x], columns=[f"feat_{i}" for i in range(len(x))])
        df_nuevo["clase"] = clase
        if os.path.exists(ruta):
            df_nuevo.to_csv(ruta, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            df_nuevo.to_csv(ruta, index=False, encoding="utf-8-sig")

        print(f"Datos corregidos guardados para la clase {clase}")



    def validacion(self, prediccion, frame_data, rutaCSV, clases):
        respuesta = input(f"¿Es correcta la predicción >>>>>>>>>'{prediccion}'? (s/n): ").strip().lower()
        if respuesta == 'n':
            clase_correcta = input("Introduce la letra correcta: ").strip().upper()
            if clase_correcta in clases:
                self.guardadDatos(frame_data, clase_correcta, rutaCSV)
            else:
                print("Letra no válida.")
            return 0
        elif respuesta == 's':
            print("Predicción confirmada.")
            return 0
        else:
            return 27
    def graficar(self, history):
        plt.figure(figsize=(10, 4))

        # Gráfico de pérdida (loss)
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Entrenamiento")
        plt.plot(history.history["val_loss"], label="Validación")
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.title("Evolución de la pérdida")
        plt.legend()

        # Gráfico de precisión (accuracy)
        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Entrenamiento")
        plt.plot(history.history["val_accuracy"], label="Validación")
        plt.xlabel("Épocas")
        plt.ylabel("Precisión")
        plt.title("Evolución de la precisión")
        plt.legend()

        # Mostrar gráficas
        plt.show()
    
    def Entrenar(self, rutaCSV, rutaRed,clases):
        if not os.path.exists(rutaCSV):
            print("No hay datos corregidos para reentrenar.")
            return

        df = pd.read_csv(rutaCSV)
        print(df.head())
        X_train = df.iloc[:, :-1].values  # Todas las características
        y_train = df.iloc[:, -1].values  # Última columna (clase)

        # Convertir etiquetas a valores numéricos
        y_train = np.array([clases.index(c) for c in y_train])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        joblib.dump(scaler, 'escalador.pkl')

        # Convertir etiquetas a one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(clases))

        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, activation="relu",  name="dense_extra1"))  
        model.add(tf.keras.layers.Dropout(0.3))  
        model.add(tf.keras.layers.Dense(len(clases), activation="softmax", name="output_extra"))

        # **Recompilar el modelo**
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # **Reentrenar modelo y guardar historial**
        print("Inicio")
        print("Valores de entrada" + str(X_train.shape))
        history = model.fit(X_train, y_train, epochs=100, verbose=1, validation_split=0.2)
        print("Finalizo")
        # **Guardar modelo**
        model.save(rutaRed)
        print("Modelo reentrenado y guardado.")

        self.graficar(history)


    def reentrenar_modelo(self, rutaCSV, rutaRed, model, scaler, clases):
        if not os.path.exists(rutaCSV):
            print("No hay datos corregidos para reentrenar.")
            return
        df = pd.read_csv(rutaCSV)
        X_train = df.iloc[:, :-1].values  # Todas las características
        y_train = df.iloc[:, -1].values  # Última columna (clase)

        # Convertir etiquetas a valores numéricos
        y_train = np.array([clases.index(c) for c in y_train])

        # Escalar características
        

        # Convertir etiquetas a one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(clases))

        # Compilar el modelo con categorical_crossentropy
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        # Reentrenar modelo
        history = model.fit(X_train, y_train, epochs=100, verbose=1, validation_split=0.2)  # Ajustar epochs según necesidad

        # Guardar modelo actualizado
        model.save(rutaRed)
        print("Modelo reentrenado y guardado.")

        self.graficar(history)