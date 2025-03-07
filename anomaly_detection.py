import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from pathlib import Path



tf.keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def load_data(filepath):
    data = pd.read_excel(filepath).interpolate(limit_area="inside").dropna()
    return data

def preprocess_data(data, scaler=None, fit_scaler=True):
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
    else:
        X_scaled = scaler.transform(data)
    return X_scaled, scaler

def reshape_for_lstm(X_scaled):
    reshaped_data = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    return reshaped_data

def build_autoencoder(input_dim, timesteps):
    input_layer = keras.Input(shape=(timesteps, input_dim))
    encoded = layers.LSTM(16, activation="relu", return_sequences=True)(input_layer)
    encoded = layers.LSTM(8, activation="relu", return_sequences=False)(encoded)
    encoded = layers.Dense(4, activation="relu")(encoded)
    decoded = layers.RepeatVector(timesteps)(encoded)
    decoded = layers.LSTM(8, activation="relu", return_sequences=True)(decoded)
    decoded = layers.LSTM(16, activation="relu", return_sequences=True)(decoded)
    decoded = layers.TimeDistributed(layers.Dense(input_dim, activation="linear"))(decoded)
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse")
    return autoencoder

def compute_mse(autoencoder, X):
    X_pred = autoencoder.predict(X, verbose=0)
    mse = np.mean(np.power(X - X_pred, 2), axis=(1, 2))
    return mse

def detect_anomalies(mse, threshold):
    return mse > threshold

def find_optimal_threshold(autoencoder, X_val):
    mse_val = compute_mse(autoencoder, X_val)
    thresholds = np.linspace(np.min(mse_val), np.max(mse_val), 100)
    best_threshold = thresholds[0]
    best_f1 = 0

    for threshold in thresholds:
        anomalies = detect_anomalies(mse_val, threshold)
        precision = np.mean(anomalies)
        recall = 1 - precision
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

def train_model(train_data, epochs, batch_size, model_path, scaler_path, logger, n_splits=5):
    train_data = load_data(train_data)
    X_scaled, scaler = preprocess_data(train_data, fit_scaler=True)
    X_reshaped = reshape_for_lstm(X_scaled)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_model = None
    best_val_loss = float('inf')
    best_threshold = None

    for fold, (train_index, val_index) in enumerate(kf.split(X_reshaped)):
        print(f"Training fold {fold + 1}/{n_splits}")
        X_train, X_val = X_reshaped[train_index], X_reshaped[val_index]

        autoencoder = build_autoencoder(input_dim=X_train.shape[2], timesteps=X_train.shape[1])
        history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                                  shuffle=True, validation_data=(X_val, X_val), verbose=1)

        val_loss = history.history['val_loss'][-1]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = autoencoder
            best_threshold = find_optimal_threshold(autoencoder, X_val)

    best_model.save(model_path)
    joblib.dump(scaler, scaler_path)
    logger.write_log(f"{Path(model_path).stem.split('_')[0]} Model Trained, best threshold: {best_threshold}")
    with open('saved_files/config.json', 'r') as f:
        diction = json.load(f)
    diction[f"{Path(model_path).stem.split('_')[0]}_threshold"] = best_threshold
    with open('saved_files/config.json', 'w') as file:
        json.dump(diction, file, indent=4)


def test_model(test_data, scaler_path, model_path):
    with open('saved_files/config.json', 'r') as f:
        config = json.load(f)
    threshold = config[f"{Path(model_path).stem.split('_')[0]}_threshold"]
    scaler = joblib.load(scaler_path)
    autoencoder = keras.models.load_model(model_path)

    X_new_scaled, _ = preprocess_data(test_data, scaler, fit_scaler=False)
    X_new_reshaped = reshape_for_lstm(X_new_scaled)
    mse_new = compute_mse(autoencoder, X_new_reshaped)
    anomalies_new = detect_anomalies(mse_new, threshold)
    anomalous_rows = test_data[anomalies_new]
    return anomalous_rows.shape[0]
