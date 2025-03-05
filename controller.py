import os
import threading
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
from logger import Logger
from keylogger import Keylogger
import smtplib
import json
from anomaly_detection import train_model, test_model


def create_path_if_not_exists(path) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def read_json() -> dict:
    with open('saved_files/config.json', 'r') as file:
        diction = json.load(file)
    if diction['email'] is not None:
        return diction

def generate_threshold(dscaler_path, dmodel_path, fscaler_path, fmodel_path):
    flight_dataframe = pd.read_excel('./saved_files/flight_data.xlsx').tail(100)
    dwell_dataframe = pd.read_excel('./saved_files/dwell_data.xlsx').tail(100)
    threshold = test_model(dwell_dataframe, dscaler_path, dmodel_path) + test_model(flight_dataframe, fscaler_path, fmodel_path)
    return threshold


def send_email(email) -> str:
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    sender_email = "ubasproject2024@gmail.com"
    sender_password = "suxc vvzo npkv faly"
    receiver_email = email
    subject = "UBAS SYSTEM::Intruder Detected"
    body = f"Unauthorized User Detected at approximate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, message.as_string())
        return f"Email sent to {email}"


def main():
    create_path_if_not_exists('saved_files')
    create_path_if_not_exists('models')
    create_path_if_not_exists('logs')

    logger = Logger()
    config = read_json()
    logger.write_log('Program Started')

    email = config['email']
    keylog = Keylogger(logger)
    keylog_thread = threading.Thread(target=keylog.start_keylogger)
    keylog_thread.start()
    last_trained_length = None
    dwell_model_path = os.path.join("auto_encoder_models", "dwell.keras")
    flight_model_path = os.path.join("auto_encoder_models", "flight.keras")
    dwell_scaler_path = os.path.join("saved_files", "dwell_scaler.pkl")
    flight_scaler_path = os.path.join("saved_files", "flight_scaler.pkl")
    threshold = None
    while True:
        cache_len = keylog.get_cache_len()
        current_length = keylog.get_length()
        if current_length >= 35000 and current_length % 500 <= 150:
            if last_trained_length is None or current_length - last_trained_length >= 500:
                logger.write_log("Training Models")
                train_model("saved_files/dwell_data.xlsx", 25, 64, dwell_model_path)
                train_model("saved_files/flight_data.xlsx", 30, 32, flight_model_path)
                threshold = generate_threshold(dwell_scaler_path, dwell_model_path, flight_scaler_path, flight_model_path)/2
                last_trained_length = current_length
        if cache_len % 100 <= 10 and cache_len >= 100:
            if current_length < 40000:
                keylog.save_dataframe()
                logger.write_log(f"Cache Added to Database.(Cache Len: {cache_len}, DB Len:{current_length})")
            elif current_length >= 40000:
                dwell_anomaly = test_model(pd.DataFrame(keylog.dwell_time, columns=['dwell_times',
                                                                               'current_key', 'last_key']), dwell_scaler_path, dwell_model_path)
                flight_anomaly = test_model(pd.DataFrame(keylog.flight_time, columns=['flight_times',
                                                                                'current_key', 'last_key']), flight_scaler_path, flight_model_path)

                if threshold is not None and (dwell_anomaly + flight_anomaly) / 2 < threshold:
                    logger.write_log(f'No Anomaly Detected: {(dwell_anomaly + flight_anomaly) / 2 - threshold} difference in scores')
                    keylog.save_dataframe()  # add to df
                    logger.write_log(f"Cache Added to Database.(Cache Len: {cache_len}, DB Len:{current_length})")
                elif threshold is not None:
                    logger.write_log(f"ANOMALY DETECTED: {(dwell_anomaly + flight_anomaly) / 2 - threshold} difference in scores")
                    keylog.flush_cache()  # flush the list
                    logger.write_log(send_email(email))  # send the email

        else:
            time.sleep(0.5)


if __name__ == "__main__":
    main()
