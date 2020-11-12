#!/usr/bin/env python
from __future__ import print_function

import cv2
import numpy as np


def resize_images(images, size):
    """
    Resizes the given images to the given size.

    Args:
        images: list|dict The opencv2 images to resize
        size: (int, int) Width and height

    Returns:
        list|dict
    """
    if type(images) is dict:
        return dict(map(lambda kv: (kv[0], cv2.resize(kv[1], size)), images.iteritems()))
    elif type(images) is list:
        return list(map(lambda image: cv2.resize(image, size), images))
    else:
        return images


font = cv2.FONT_HERSHEY_SIMPLEX


def add_topic_mark(images):
    """
    Adds the topic as string to the images

    Returns:
        dict: The images with the topic as a mark
    """
    return dict(
        map(
            lambda kv: (kv[0], cv2.putText(
                kv[1], kv[0], (5, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA
            )),
            images.iteritems()
        )
    )


class LayoutEntry:
    def __init__(self, topic, x, y):
        self.topic = topic
        self.x = int(x)
        self.y = int(y)


def layout_images(images, layout_entries):
    """
    Layouts the given images based on their layout entry.

    Args:
        images: dict The images by topic
        layout_entries: list[LayoutEntry] The layout entries defining the layout

    Returns:

    """
    max_x = max(list(map(lambda entry: entry.x, layout_entries)))
    max_y = max(list(map(lambda entry: entry.y, layout_entries)))
    if len(images) == 0:
        return np.zeros((100, 100, 3), np.uint8)

    empty_image = np.ones(images[list(images)[0]].shape, np.uint8) * 150

    rows = None
    for y in range(max_y + 1):
        row = []
        for x in range(max_x + 1):
            result = [entry for entry in layout_entries if entry.x == x and entry.y == y]
            if result:
                row.append(result[0].topic)
            else:
                row.append(None)

        row = map(lambda topic: images[topic] if topic in images.keys() and topic is not None else empty_image, row)
        row = np.concatenate(row, axis=1)
        if rows is None:
            rows = row
        else:
            rows = np.concatenate((rows, row))

    return rows
