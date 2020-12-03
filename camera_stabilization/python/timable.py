from abc import ABC
from collections import OrderedDict
from datetime import datetime


class ITimable:
    line_format = "[{}] {}: {}\n"

    def __init__(self, name: str = None):
        self.timestamps = OrderedDict()
        self.add_timestamp('construction')
        if name:
            self.name = name
        else:
            self.name = 'Unnamed'

        self.sequence_num = 0

    def clear_timestamps(self):
        self.timestamps = OrderedDict()
        self.add_timestamp('construction')

    def add_timestamp(self, name: str = None):
        self.timestamps[datetime.now()] = name

    def get_durations(self):
        times = list(self.timestamps.keys())
        names = list(self.timestamps.values())
        return {names[i]: (times[i] - times[i - 1]).total_seconds() for i in range(1, len(times))}

    def get_latest_duration(self):
        durations = self.get_durations()
        if len(durations) <= 0:
            return -1
        return durations[list(durations.keys())[-1]]

    def get_timestamps(self):
        times = list(self.timestamps.keys())
        names = list(self.timestamps.values())
        return {names[i]: times[i] for i in range(1, len(times))}

    def to_str(self, durations = True, reset = True):
        if durations:
            values = self.get_durations()
        else:
            values = self.get_timestamps()
        timings = ""
        for timestamp in values.items():
            timings += ITimable.line_format.format(self.name, timestamp[1], timestamp[0])
        if reset:
            self.clear_timestamps()
        return timings
