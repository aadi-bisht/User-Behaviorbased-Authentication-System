import os

import joblib
from tensorboard.compat.tensorflow_stub.io.gfile import exists

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from anomaly_detection import reshape_for_lstm, load_data, preprocess_data, detect_anomalies, compute_mse, build_autoencoder, train_model, test_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path


def train_and_evaluate_incremental(data_path, model_name, epochs, batch_size, scaler_path, increments=8):
    full_data = load_data(data_path)
    # X_full_scaled, scaler = preprocess_data(full_data, scaler, fit_scaler=False)
    # X_full_reshaped = reshape_for_lstm(X_full_scaled)

    X_train, X_test = train_test_split(full_data, test_size=100)

    increment_size = len(X_train) // increments

    train_sizes = []

    anomaly_counts = []

    for i in range(1, increments + 1):
        print(f"Training on {i * increment_size} samples...")

        X_train_subset = X_train[:i * increment_size]
        train_model(X_train_subset, epochs, batch_size, f"test/{model_name}test.keras", scaler_path)
        #
        # autoencoder = build_autoencoder(input_dim=X_train_subset.shape[2], timesteps=X_train_subset.shape[1])
        # autoencoder.fit(X_train_subset, X_train_subset, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)
        #
        # mse_train = compute_mse(autoencoder, X_train_subset)
        # threshold = round(0.2 * np.mean(mse_train), 7)
        #
        # mse_test = compute_mse(autoencoder, X_test)
        # anomalies = detect_anomalies(mse_test, threshold)
        # anomaly_count = np.sum(anomalies)
        anomaly_count = test_model(X_test, scaler_path, f"test/{model_name}test.keras")
        train_sizes.append(i * increment_size)
        anomaly_counts.append(anomaly_count)

        print(f"Number of anomalies detected: {anomaly_count}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, anomaly_counts, marker='o')
    plt.title(f'Anomaly Detection Performance vs Training Data Size ({model_name})')
    plt.xlabel('Training Data Size')
    plt.ylabel('Number of Anomalies Detected')
    plt.grid(True)
    plt.savefig(f'saved_files/{model_name}_performance.png')
    plt.close()


os.makedirs('test', exist_ok=True)
dwell_scaler_path = os.path.join("saved_files", "dwell_scaler.pkl")
flight_scaler_path = os.path.join("saved_files", "flight_scaler.pkl")
train_and_evaluate_incremental("saved_files/dwell_data.xlsx", "dwell", epochs=25, batch_size=64, scaler_path=dwell_scaler_path)
train_and_evaluate_incremental("saved_files/flight_data.xlsx", "flight", epochs=30, batch_size=32, scaler_path=flight_scaler_path)
