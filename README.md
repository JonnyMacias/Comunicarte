# ✋🤖 Reconocimiento de Señales en Tiempo Real usando LSTM

Este proyecto tiene como objetivo identificar en **tiempo real** la **señal mostrada con las manos** frente a una cámara, utilizando un modelo de red neuronal LSTM previamente entrenado.

---

## 📌 Descripción general del flujo

1. 📦 **Recolección del Dataset**  
   Se capturan imágenes y videos correspondientes a cada seña del lenguaje gestual. Estas se almacenan como base para el entrenamiento.

2. 🔍 **Extracción de características**  
   Para cada imagen o frame de video:
   - Se identifican los puntos clave de las manos.
   - Se calculan las **pendientes** y **ángulos** entre estos puntos.
   - Los datos resultantes se agrupan según la seña correspondiente.

3. 🧾 **Almacenamiento en CSV**  
   Toda la información procesada (pendientes y ángulos) se guarda en archivos `.csv`, organizados por clase (una clase por seña).

4. 🧠 **Entrenamiento del modelo LSTM**  
   Se utiliza una red neuronal **LSTM (Long Short-Term Memory)** para aprender la secuencia de movimientos y formas de las manos correspondientes a cada seña.

5. 🟢 **Reconocimiento en tiempo real**  
   Finalmente, con el modelo ya entrenado:
   - Se activa la cámara.
   - Se detectan las manos en vivo.
   - Se calculan en tiempo real las pendientes y ángulos.
   - Se hace una predicción utilizando el modelo LSTM para identificar qué seña se está mostrando.

---

## 🛠️ Tecnologías utilizadas

- **Python**
- **OpenCV**
- **MediaPipe** (para la detección de manos y puntos clave)
- **NumPy / Pandas**
- **TensorFlow / Keras** (para construir y entrenar el modelo LSTM)
