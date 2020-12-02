from abc import ABC
from datetime import datetime


class ITimable(ABC):
    line_format = "[{}] {}: {}\n"

    def __init__(self, name: str = None):
        self.timestamps = []
        if name:
            self.name = name
        else:
            self.name = 'Unnamed'

    def clear_timestamps(self):
        self.timestamps = []

    def add_timestamp(self, name: str):
        self.timestamps.append((name, datetime.now()))

    def timestamps_to_str(self, reset=False):
        timings = ""
        for timestamp in self.timestamps:
            timings += ITimable.line_format.format(self.name, timestamp[0], timestamp[1])
        if reset:
            self.clear_timestamps()
        return timings

    def durations_to_str(self, reset=False):
        timings = ""
        for i in range(1, len(self.timestamps)):
            timings += ITimable.line_format.format(self.name, self.timestamps[i][0],
                                                   self.timestamps[i][1] - self.timestamps[i - 1][1])
        timings += ITimable.line_format.format(self.name, 'total',
                                               self.timestamps[-1][1] - self.timestamps[0][1])
        if reset:
            self.clear_timestamps()
        return timings
