import time
import pandas
import keyboard

LIMIT = 42500

def is_key_event(string) -> bool:
    keywords = ['alt', 'alt gr', 'ctrl', 'left alt', 'left ctrl', 'left shift', 'left windows', 'right alt',
                'right ctrl', 'right shift', 'right windows', 'shift', 'windows', 'enter', 'space', 'tab',
                'insert', 'caps lock', 'esc', 'left', 'right', 'up', 'down', 'backspace', 'print screen', 'delete',
                'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12']
    if string in keywords:
        return False
    else:
        return True


def _update_df(df):
    return df.drop(index=df.index[:(len(df) - LIMIT)], inplace=True)


class Keylogger:
    def __init__(self, logger):
        self.dwell_time = []
        self.flight_time = []
        self.dwell_start_time = None
        self.flight_start_time = None
        self.running = True
        self.last_key = "~"
        self.flight_dataframe = pandas.DataFrame()
        self.dwell_dataframe = pandas.DataFrame()
        self.logger = logger

    def get_length(self) -> int:
        return min(len(self.dwell_dataframe), len(self.flight_dataframe))

    def get_cache_len(self) -> int:
        return min(len(self.dwell_time), len(self.flight_time))

    def record_keys(self, event) -> None:
        try:
            if event.event_type == keyboard.KEY_UP and self.dwell_start_time is None:
                return
            current_time = event.time

            if event.event_type == keyboard.KEY_DOWN :
                self.dwell_start_time = current_time
                if self.flight_start_time is not None and is_key_event(event.name):
                    f_time = current_time - self.flight_start_time
                    if 0 < f_time <= 1:
                        self.flight_time.append([f_time*10, ord(event.name), ord(self.last_key)])

            elif event.event_type == keyboard.KEY_UP and is_key_event(event.name):
                self.flight_start_time = current_time
                if self.dwell_start_time is not None:
                    d_time = current_time - self.dwell_start_time
                    if d_time > 0:
                        self.dwell_time.append([d_time*10, ord(event.name), ord(self.last_key)])
                    self.last_key = event.name

        except TypeError:
            self.dwell_start_time = None
            self.flight_start_time = None

    def flush_cache(self) -> None:
        self.dwell_time.clear()
        self.flight_time.clear()

    def start_keylogger(self) -> None:
        self.load_dataframe()
        self.running = True
        keyboard.hook(self.record_keys)
        while self.running:
            time.sleep(0.1)

    def stop_keylogger(self) -> None:
        self.running = False
        keyboard.unhook_all()

    def convert_to_df(self) -> None:
        df_dwell = pandas.DataFrame({
            'dwell_times': [entry[0] for entry in self.dwell_time],
            'current_key': [entry[1] for entry in self.dwell_time],
            'last_key': [entry[2] for entry in self.dwell_time]
        })

        df_flight = pandas.DataFrame({
            'flight_times': [entry[0] for entry in self.flight_time],
            'current_key': [entry[1] for entry in self.flight_time],
            'last_key': [entry[2]for entry in self.flight_time]
        })

        if self.flight_dataframe.empty:
            self.flight_dataframe = df_flight
        else:
            self.flight_dataframe = pandas.concat([self.flight_dataframe, df_flight], ignore_index=True)

        if self.dwell_dataframe.empty:
            self.dwell_dataframe = df_dwell
        else:
            self.dwell_dataframe = pandas.concat([self.dwell_dataframe, df_dwell], ignore_index=True)

    def save_dataframe(self) -> None:
        self.convert_to_df()
        if self.get_length() >= LIMIT:
            self.dwell_dataframe = _update_df(self.dwell_dataframe)
            self.flight_dataframe = _update_df(self.flight_dataframe)

        self.flight_dataframe.to_excel('./saved_files/flight_data.xlsx', index=False)
        self.dwell_dataframe.to_excel('./saved_files/dwell_data.xlsx', index=False)
        self.flush_cache()
        # self.flight_dataframe.to_excel('./saved_files/unauthorized_flight_data_1.xlsx', index=False)
        # self.dwell_dataframe.to_excel('./saved_files/unauthorized_dwell_data_1.xlsx', index=False)


    def load_dataframe(self) -> None:
        try:
            self.flight_dataframe = pandas.read_excel('./saved_files/flight_data.xlsx')
            self.dwell_dataframe = pandas.read_excel('./saved_files/dwell_data.xlsx')
            # self.flight_dataframe = pandas.read_excel('./saved_files/unauthorized_flight_data_1.xlsx')
            # self.dwell_dataframe = pandas.read_excel('./saved_files/unauthorized_dwell_data_1.xlsx')

        except FileNotFoundError:
            pass

    def print_times(self) -> None:
        print("Dwell times: ", self.dwell_time, "Count: ", len(self.dwell_time))
        print("Flight times: ", self.flight_time, "Count: ", len(self.flight_time))

    def __repr__(self) -> str:
        if not self.dwell_dataframe.empty and not self.flight_dataframe.empty:
            return (f"Flight Dataframe: {self.flight_dataframe.tail(50), self.flight_dataframe.shape[0]}\n"
                    f"Dwell Dataframe: {self.dwell_dataframe.tail(50), self.dwell_dataframe.shape[0]}")
        else:
            return f"Dataframe: {self.flight_dataframe, self.dwell_dataframe}"
