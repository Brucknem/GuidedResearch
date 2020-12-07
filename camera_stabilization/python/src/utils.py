import random
from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
from scipy.signal import argrelextrema


class FixedSizeDataStructure(ABC):
    """
    Abstract base class for all fixed size data structures.
    """

    def __init__(self, max_num_elements: int, remove_random: bool):
        """
        constructor

        :param max_num_elements: The maximum number of elements in the data structure.
        :param remove_random: Flag if to remove elements from the beginning or random.
        """
        self._max_num_elements = max_num_elements
        self._remove_random = remove_random

    def is_fixed_size(self):
        """
        Has the data structure a fixed size.

        :return: True if the size is fixed, False else
        """
        return self._max_num_elements > 0

    def remove_if_necessary(self):
        """
        Removes as much elements as needed to not exceed the fixed size.
        """
        if self.is_fixed_size():
            while len(self.get_data_structure()) > self._max_num_elements:
                del self.get_data_structure()[self.get_element_to_delete()]

    def get_element_to_delete(self):
        """
        Returns the first or a random element (excluding the latest).

        :return: The element
        """
        if self._remove_random:
            return self.get_element_to_delete_random()
        return self.get_element_to_delete_front()

    @abstractmethod
    def get_element_to_delete_front(self):
        """
        Getter for the first element in the underlying data structure.

        :return: The first element
        """
        pass

    @abstractmethod
    def get_element_to_delete_random(self):
        """
        Getter for a random element (excluding the latest) in the underlying data structure.

        :return: A random element (excluding the latest)
        """
        pass

    @abstractmethod
    def get_data_structure(self):
        """
        Getter for the underlying data structure.
        :return:
        """
        pass


class FixedSizeOrderedDict(FixedSizeDataStructure, OrderedDict):
    """
    An ordered dictionary with a (possibly) fixed size
    """

    def __init__(self, max_num_elements: int = 0, remove_random: bool = False, *args, **kwargs):
        FixedSizeDataStructure.__init__(self, max_num_elements, remove_random)
        OrderedDict.__init__(self, *args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self.remove_if_necessary()

    def get_element_to_delete_front(self):
        return list(self.keys())[0]

    def get_element_to_delete_random(self):
        return random.choice(list(self.keys())[:-1])

    def get_data_structure(self):
        return self


class FixedSizeSortedDict(FixedSizeOrderedDict):
    def __init__(self, max_num_elements: int = 0, remove_random: bool = False, *args, **kwargs):
        super().__init__(max_num_elements, remove_random, *args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self = sorted(self)


class FixedSizeList(FixedSizeDataStructure, list):
    """
    A list with a (possibly) fixed size
    """

    def __init__(self, max_num_elements: int = 0, remove_random: bool = False, *args, **kwargs):
        FixedSizeDataStructure.__init__(self, max_num_elements, remove_random)
        list.__init__(self, *args, **kwargs)

    def append(self, value: any) -> None:
        super().append(value)
        self.remove_if_necessary()

    def __setitem__(self, value, **kwargs):
        list.__setitem__(self, value, **kwargs)
        self.remove_if_necessary()

    def get_element_to_delete_front(self):
        return 0

    def get_element_to_delete_random(self):
        return random.randint(0, self.__len__() - 1)

    def get_data_structure(self):
        return self
