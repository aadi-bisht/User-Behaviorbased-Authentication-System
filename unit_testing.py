import smtplib
import unittest
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock, call
from controller import create_path_if_not_exists, read_json, send_email
from logger import Logger, get_time, make_log_file
from keylogger import Keylogger, is_key_event, _update_df
from anomaly_detection import load_data, preprocess_data
from setup import install_and_import

class ControllerTestCases(unittest.TestCase):
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    def test_create_path_if_not_exists(self, mock_exists, mock_makedirs):
        create_path_if_not_exists("test_path")
        mock_makedirs.assert_called_once_with("test_path")

    @patch("os.path.exists", return_value=True)
    def test_create_path_if_exists(self, mock_exists):
        with patch("os.makedirs") as mock_makedirs:
            create_path_if_not_exists("test_path")
            mock_makedirs.assert_not_called()

    @patch("builtins.open", new_callable=mock_open, read_data='{"email": "test@example.com"}')
    def test_read_json_valid(self, mock_file):
        data = read_json()
        self.assertEqual(data, {"email": "test@example.com"})

    @patch("builtins.open", new_callable=mock_open, read_data='{}')
    def test_read_json_missing_email(self, mock_file):
        with self.assertRaises(KeyError):
            read_json()

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_read_json_file_not_found(self, mock_file):
        with self.assertRaises(FileNotFoundError):
            read_json()

    @patch("smtplib.SMTP")
    def test_send_email_success(self, mock_smtp):
        mock_server = mock_smtp.return_value.__enter__.return_value
        result = send_email("test@example.com")
        mock_server.sendmail.assert_called_once()
        self.assertIn("Email sent to test@example.com", result)

    @patch("smtplib.SMTP", side_effect=smtplib.SMTPException("SMTP Error"))
    def test_send_email_failure(self, mock_smtp):
        with self.assertRaises(smtplib.SMTPException):
            send_email("test@example.com")


class TestLogger(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open)
    def test_write_log(self, mock_file):
        logger = Logger()
        logger.write_log("Test message")
        mock_file.assert_called_with("logs/log_file.txt", "a+")

    def test_get_time_format(self):
        time_str = get_time()
        self.assertRegex(time_str, r"\(\d{2}/\d{2}/\d{4}\)\d{2}:\d{2}:\d{2}:: ")

    @patch("os.path.exists", return_value=False)
    @patch("builtins.open", new_callable=mock_open)
    def test_make_log_file_not_exists(self, mock_file, mock_exists):
        make_log_file()
        mock_file.assert_called_once()
        mock_file().write.assert_called_once()

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    def test_make_log_file_exists(self, mock_file, mock_exists):
        make_log_file()
        mock_file.assert_not_called()

    @patch("logger.make_log_file")
    def test_logger_init(self, mock_make_log_file):
        logger = Logger()
        mock_make_log_file.assert_called_once()
        self.assertEqual(logger.file_name, "logs/log_file.txt")


class TestKeylogger(unittest.TestCase):
    def setUp(self):
        self.logger = MagicMock()
        self.keylogger = Keylogger(self.logger)

    def test_get_length(self):
        self.assertEqual(self.keylogger.get_length(), 0)

    def test_is_key_event(self):
        self.assertTrue(is_key_event("a"))
        self.assertTrue(is_key_event("1"))
        self.assertTrue(is_key_event("z"))

        self.assertFalse(is_key_event("alt"))
        self.assertFalse(is_key_event("ctrl"))
        self.assertFalse(is_key_event("shift"))
        self.assertFalse(is_key_event("enter"))
        self.assertFalse(is_key_event("f1"))

    def test_update_df(self):
        test_df = pd.DataFrame({'col': range(50000)})
        original_len = len(test_df)

        _update_df(test_df)

        self.assertEqual(len(test_df), 42500)

    def test_get_cache_len(self):
        self.keylogger.dwell_time = [[0.1, 97, 98], [0.2, 98, 99]]
        self.keylogger.flight_time = [[0.05, 97, 98]]

        self.assertEqual(self.keylogger.get_cache_len(), 1)

    def test_flush_cache(self):
        self.keylogger.dwell_time = [[0.1, 97, 98], [0.2, 98, 99]]
        self.keylogger.flight_time = [[0.05, 97, 98]]

        self.keylogger.flush_cache()

        self.assertEqual(len(self.keylogger.dwell_time), 0)
        self.assertEqual(len(self.keylogger.flight_time), 0)

    @patch("pandas.read_excel")
    def test_load_dataframe_success(self, mock_read_excel):
        mock_read_excel.side_effect = [
            pd.DataFrame({'flight_times': [0.05], 'current_key': [97], 'last_key': [98]}),
            pd.DataFrame({'dwell_times': [0.1], 'current_key': [97], 'last_key': [98]})
        ]

        self.keylogger.load_dataframe()

        mock_read_excel.assert_has_calls([
            call('./saved_files/flight_data.xlsx'),
            call('./saved_files/dwell_data.xlsx')
        ])

    @patch("pandas.read_excel", side_effect=FileNotFoundError)
    def test_load_dataframe_file_not_found(self, mock_read_excel):
        self.keylogger.load_dataframe()

        mock_read_excel.assert_called_once()


class TestAnomalyDetection(unittest.TestCase):
    @patch("pandas.read_excel", return_value=pd.DataFrame({"col": [1, 2, 3]}))
    def test_load_data(self, mock_read):
        data = load_data("test.xlsx")
        self.assertFalse(data.empty)

    def test_preprocess_data(self):
        data = pd.DataFrame({"col": [1, 2, 3]})
        X_scaled, scaler = preprocess_data(data)
        self.assertIsNotNone(scaler)


class TestSetup(unittest.TestCase):
    @patch("subprocess.check_call")
    def test_install_and_import(self, mock_subproc):
        install_and_import("fakepackage")
        mock_subproc.assert_called()


if __name__ == '__main__':
    unittest.main()
