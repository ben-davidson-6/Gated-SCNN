import numpy as np


def random_image(h, w):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def random_label(h, w, classes, flat=True):
    seg = np.random.randint(0, classes, (h, w), dtype=np.uint8)
    if not flat:
        seg = np.eye(classes)[seg]
    return seg


def random_edge(h, w, flat=True):
    return random_label(h, w, classes=1, flat=flat)
