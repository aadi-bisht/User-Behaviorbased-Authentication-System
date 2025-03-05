import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from pathlib import Path
import os

# Setting random seed for reproducibility across multiple runs
tf.keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism()

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
        precision = np.mean(anomalies)  # Assuming all validation samples are normal
        recall = 1 - precision
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

def train_model(train_data, epochs, batch_size, model_path, scaler_path=None, n_splits=5):
    train_data = load_data(train_data)
    X_scaled, scaler = preprocess_data(train_data, fit_scaler=True)
    X_reshaped = reshape_for_lstm(X_scaled)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_model = None
    best_val_loss = float('inf')
    best_threshold = None


    # X_train, X_test = train_test_split(X_train_reshaped, test_size=0.2)
    # autoencoder = build_autoencoder(input_dim=X_train.shape[2], timesteps=X_train.shape[1])
    # autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test), verbose=1)
    # autoencoder.save(model_path)

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
    # mse_train = compute_mse(autoencoder, X_test)
    # threshold = round(0.2 * np.mean(mse_train), 7)
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
    print(f"{Path(model_path).stem.split('_')[0]} Number of anomalies detected: {anomalous_rows.shape[0]}")
    return anomalous_rows.shape[0]

# def main():
#     os.makedirs('auto_encoder_models', exist_ok=True)
#     dwell_model_path = os.path.join("auto_encoder_models", "dwell.keras")
#     flight_model_path = os.path.join("auto_encoder_models", "flight.keras")
#     dwell_scaler_path = os.path.join("saved_files", "dwell_scaler.pkl")
#     flight_scaler_path = os.path.join("saved_files", "flight_scaler.pkl")
#     train_model("saved_files/dwell_data.xlsx", 25, 64, dwell_model_path, dwell_scaler_path)
#     train_model("saved_files/flight_data.xlsx", 30, 32, flight_model_path, flight_scaler_path)

#     print("Original User Results:")
#     test_model(load_data("saved_files/dwell_data.xlsx").tail(100), dwell_scaler_path, dwell_model_path)
#     test_model(load_data("saved_files/flight_data.xlsx").tail(100), flight_scaler_path, flight_model_path)
#     print("User1 Results:")
#     test_model(load_data("saved_files/unauthorized_dwell_data_1.xlsx").tail(100), dwell_scaler_path, dwell_model_path)
#     test_model(load_data("saved_files/unauthorized_flight_data_1.xlsx").tail(100), flight_scaler_path, flight_model_path)
#     print("User2 Results:")
#     test_model(load_data("saved_files/unauthorized_dwell_data_2.xlsx").tail(100), dwell_scaler_path, dwell_model_path)
#     test_model(load_data("saved_files/unauthorized_flight_data_2.xlsx").tail(100), flight_scaler_path, flight_model_path)
#     print("User3 Results:")
#     test_model(load_data("saved_files/unauthorized_dwell_data_3.xlsx").tail(100), dwell_scaler_path, dwell_model_path)
#     test_model(load_data("saved_files/unauthorized_flight_data_3.xlsx").tail(100), flight_scaler_path, flight_model_path)
#     print("User4 Results:")
#     test_model(load_data("saved_files/unauthorized_dwell_data_4.xlsx").tail(100), dwell_scaler_path, dwell_model_path)
#     test_model(load_data("saved_files/unauthorized_flight_data_4.xlsx").tail(100), flight_scaler_path, flight_model_path)
#
#
# if __name__ == "__main__":
#     main()
