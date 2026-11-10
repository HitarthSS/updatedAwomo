"""Locate toolbox tools in head RGB with appearance exemplars from development."""
from dataclasses import dataclass

import cv2
import numpy as np


# Pixel crops and a shaft-aligned grasp point on the layout 0 development image.
# The execution scene is searched anew. These are appearance templates.
EXEMPLARS = {
    "hammer": ((141, 197, 182, 226), (149, 246)),
    "pliers": ((478, 189, 512, 229), (495, 226)),
    "tape measure": ((412, 199, 451, 236), (429, 218)),
    "wrench": ((209, 180, 240, 208), (215, 225)),
}


@dataclass(frozen=True)
class ToolLocation:
    center: np.ndarray
    grasp: np.ndarray
    opening: np.ndarray
    score: float


class ToolLocalizer:
    def __init__(self, development_rgb):
        reference = cv2.cvtColor(development_rgb, cv2.COLOR_RGB2GRAY)
        self.templates = {}
        for name, ((x0, y0, x1, y1), grasp) in EXEMPLARS.items():
            patch = reference[y0:y1, x0:x1]
            h, w = patch.shape
            center = np.asarray([w / 2, h / 2])
            offset = np.asarray(grasp) - [x0, y0] - center
            self.templates[name] = self._variants(patch, center, offset)

    @staticmethod
    def _variants(patch, center, offset):
        variants = []
        for scale in (.85, .95, 1.05, 1.15):
            for angle in (-24, -12, 0, 12, 24):
                transform = cv2.getRotationMatrix2D(center, angle, scale)
                warped = cv2.warpAffine(patch, transform, patch.shape[::-1], borderMode=cv2.BORDER_REFLECT_101)
                variants.append((warped, center, transform[:, :2] @ offset))
        return variants

    def locate(self, rgb, name, near=None):
        if rgb.shape[:2] != (480, 640):
            raise ValueError("The calibrated toolbox localizer expects 640×480 RGB")
        # The released ARX-X5 toolbox task starts the dark tools on the left,
        # and the yellow tools on the right. Shape matching resolves identity.
        x, end = (75, 277) if name in ("hammer", "wrench") else (381, 564)
        y = 125
        bottom = 270
        if near is not None:
            u, v = np.rint(near).astype(int)
            x, end = max(x, u - 38), min(end, u + 38)
            y, bottom = max(y, v - 38), min(bottom, v + 38)
        view = cv2.cvtColor(rgb[y:bottom, x:end], cv2.COLOR_RGB2GRAY)
        best = None
        for template, center, offset in self.templates[name]:
            response = cv2.matchTemplate(view, template, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, upper_left = cv2.minMaxLoc(response)
            if best is None or confidence > best.score:
                found = np.asarray(upper_left, dtype=float) + (x, y) + center
                # Fingers open across the measured shaft. The compact tape
                # measure uses an x-directed pinch in the head image.
                across = np.asarray([offset[1], -offset[0]])
                if name == "tape measure" or np.linalg.norm(across) < 1:
                    across = np.asarray([1., 0.])
                best = ToolLocation(found, found + offset,
                                    across / np.linalg.norm(across), float(confidence))
        return best if best.score >= .70 else None
