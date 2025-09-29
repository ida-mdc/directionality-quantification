import unittest

import numpy as np

from directionality_quantification.main import orientation_with_anisotropy_from_root


def make_blank(h=64, w=64):
    return np.zeros((h, w), dtype=bool)


def draw_hline(img, y, x0, x1):
    img[y, min(x0, x1):max(x0, x1) + 1] = True


class TestCellExtensionOrientation(unittest.TestCase):

    def test_one_sided_line(self):
        img = make_blank()
        draw_hline(img, y=32, x0=10, x1=50)
        # Choose a root near the left end (like “biggest circle center + offset”)
        root = (10, 32)
        aniso, v, L = orientation_with_anisotropy_from_root(img, root)
        assert aniso > 0.9, f"anisotropy too low: {aniso}"
        expected = 50 - 10
        assert abs(L - expected) <= 2, f"net length {L} != ~{expected}"

    def test_opposite_arms_cancel(self):
        img = make_blank()
        draw_hline(img, y=32, x0=16, x1=32)
        draw_hline(img, y=32, x0=32, x1=48)
        # Root at the junction pixel (x=32,y=32)
        root = (32, 32)
        aniso, v, L = orientation_with_anisotropy_from_root(img, root)
        assert aniso > 0.9, f"anisotropy too low: {aniso}"
        assert L < 2.0, f"opposite arms should cancel; got length {L}"

    def test_unbalanced_arms_partial(self):
        img = make_blank()
        draw_hline(img, y=32, x0=20, x1=32)   # left arm 12
        draw_hline(img, y=32, x0=32, x1=56)   # right arm 24
        # Root at the junction
        root = (32, 32)
        aniso, v, L = orientation_with_anisotropy_from_root(img, root)
        assert aniso > 0.7
        expected = (56 - 32) - (32 - 20)  # 24 - 12 = 12
        assert abs(L - expected) <= 2, f"net length {L} != ~{expected}"
