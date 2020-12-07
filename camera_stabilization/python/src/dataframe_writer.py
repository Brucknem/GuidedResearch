import os
from datetime import datetime
from threading import Thread

import pandas as pd
import numpy as np

from pathlib import Path


class Dataframe(dict):
    """
    A dictionary with lists as values.
    The value lists are length-synchronized, so when adding a new or incomplete row
    the other rows are filled with np.NaN values until all are of equal length.
    """

    def append_row(self, **columns):
        """
        Appends a row to the dataframe.
        Synchronizes the lengths of the other columns by adding np.NaN values.
        Fills new columns with np.NaN values until the current length.

        :param columns: Key, value mapping of columns
        """
        length = len(self)

        for entry in columns.items():
            if entry[0] not in self:
                self[entry[0]] = [np.NaN] * length
            self[entry[0]].append(entry[1])

        for key in self.keys():
            difference = length - len(self[key]) + 1
            if difference > 0:
                self[key].append([np.NaN] * difference)

    def __len__(self) -> int:
        """
        Gets the maximum length of the columns.

        :return: The length
        """
        number_elements = 0
        for column_values in self.values():
            length = len(column_values)
            if length > number_elements:
                number_elements = length
        return number_elements

    def to_pandas(self) -> pd.DataFrame:
        """
        Converts the dataframe to a pandas dataframe.
        """
        result = pd.DataFrame.from_dict(self)
        return result


class DataframeWriter:
    """
    A writer for dataframes.
    Holds a dict of named dataframes to which new values can be appended.
    """

    COMMON_COLUMNS = ['Milliseconds', 'Timestamp']
    MAXIMA_COLUMN = ['Maxima']

    def __init__(self, base_path, write_every_n_seconds: int = 3):
        """
        constructor

        :param base_path: The output path of the dataframe files.
        :param write_every_n_seconds:
        """
        self.dataframes = {}
        self.base_path = os.path.expanduser(base_path)
        Path(self.base_path).mkdir(exist_ok=True, parents=True)
        self.last_write = datetime.now()
        self.write_every_n_seconds = write_every_n_seconds

    def write(self):
        """
        Write all dataframes to disk.
        """
        dataframes = dict(self.dataframes.items())
        for dataframe in dataframes.items():
            dataframe[1].to_pandas().to_csv(
                os.path.join(self.base_path, '{}.csv'.format(dataframe[0])), index=False
            )

    def append(self, name, **columns):
        """
        Appends the columns to the dataframe with the given name.

        :param name: The name of the dataframe
        :param columns: The columns and values to append
        """
        start_write = datetime.now()
        name = str(name)
        if name not in self.dataframes:
            self.dataframes[name] = Dataframe()
            last_write = start_write
        else:
            timestamp_column = self.dataframes[name][DataframeWriter.COMMON_COLUMNS[1]]
            last_write = timestamp_column[0]

        milliseconds = (start_write - last_write).total_seconds() * 1000
        new_values = dict(zip(DataframeWriter.COMMON_COLUMNS, [milliseconds, start_write]))
        new_values = {**new_values, **columns}
        self.dataframes[name].append_row(**new_values)

        if (start_write - self.last_write).total_seconds() > self.write_every_n_seconds:
            self.last_write = start_write
            Thread(target=self.write).start()
            # self.write()
