import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Masking, concatenate
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

# ------------------ Cargar datos ------------------
X_secuencia = []  # 83x90 = 7470
X_fourier = []    # 320
y = []

with open('resources/limpio.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # saltar encabezado

    for row in reader:
        row_f = [float(x) for x in row[:-1]]
        X_secuencia.append(row_f[:3600])   # primeros 7470 valores
        X_fourier.append(row_f[3600:3600+320])  # últimos 320
        y.append(row[-1])

X_secuencia = np.array(X_secuencia, dtype=np.float32)
X_fourier = np.array(X_fourier, dtype=np.float32)

# ------------------ Etiquetas ------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)

# ------------------ Escalar ------------------
scaler_seq = StandardScaler()
scaler_fourier = StandardScaler()

X_seq_scaled = scaler_seq.fit_transform(X_secuencia)
X_fourier_scaled = scaler_fourier.fit_transform(X_fourier)

joblib.dump(scaler_seq, 'resources/scaler_seq.pkl')
joblib.dump(scaler_fourier, 'resources/scaler_fourier.pkl')
joblib.dump(label_encoder, 'resources/label_encoder.pkl')

# ------------------ Reestructurar secuencia ------------------
n_timesteps = 40
n_features = 90
X_seq_scaled = X_seq_scaled.reshape((-1, n_timesteps, n_features))

# ------------------ Dividir ------------------
X_seq_train, X_seq_test, X_fourier_train, X_fourier_test, y_train, y_test = train_test_split(
    X_seq_scaled, X_fourier_scaled, y, test_size=0.2, random_state=42)

# ------------------ Red con doble entrada ------------------
input_seq = Input(shape=(n_timesteps, n_features))
x = Masking(mask_value=0.0)(input_seq)
x = LSTM(64, return_sequences=True)(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = LSTM(32, return_sequences=True)(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = LSTM(16)(x)
x = Dropout(0.2)(x)

input_fourier = Input(shape=(320,))
yf = Dense(128, activation='relu')(input_fourier)
yf = Dropout(0.3)(yf)

combined = concatenate([x, yf])
z = Dense(64, activation='relu')(combined)
z = Dropout(0.3)(z)
out = Dense(y.shape[1], activation='softmax')(z)

model = Model(inputs=[input_seq, input_fourier], outputs=out)

model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit([X_seq_train, X_fourier_train], y_train, epochs=150, batch_size=16,
          validation_data=([X_seq_test, X_fourier_test], y_test), callbacks=[early_stopping])

# ------------------ Guardar ------------------
model.save('resources/Letras_LSTM_doble_entrada.keras', save_format='keras')
print("Modelo LSTM con Fourier entrenado y guardado.")
print("Clases:", label_encoder.classes_)
