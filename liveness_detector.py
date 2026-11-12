"""
Face liveness (anti-spoofing) detector.

A MobileNetV2 backbone with a small classification head that outputs a
spoof score between 0 (live face) and 1 (fake face: printed photo, replayed
video, picture on a screen, ...).
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(BASE_DIR, "weights", "liveness.hdf5")
DEFAULT_THRESHOLD = 0.10
INPUT_SIZE = 224
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class LivenessResult:
    """Prediction for one image."""

    score: float    # spoof score: 0 = live, 1 = fake
    is_live: bool

    @property
    def label(self) -> str:
        return "Live" if self.is_live else "Fake"


def _create_network() -> Model:
    """Builds the network architecture (weights are loaded separately)."""
    backbone = MobileNetV2(input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
                           include_top=False,
                           weights=None)

    features = Conv2D(32, (3, 3), activation="relu")(backbone.output)
    features = Dropout(rate=0.2, name="extra_dropout1")(features)
    features = GlobalAveragePooling2D()(features)
    output = Dense(1, activation="sigmoid", name="classifier")(features)

    return Model(inputs=backbone.input, outputs=output, name="mobilenetv2_spoof")


class LivenessDetector:
    """Loads the model once and classifies images as live or fake."""

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS,
                 threshold: float = DEFAULT_THRESHOLD):
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        self.threshold = threshold
        self.model = _create_network()
        self.model.load_weights(weights_path)

    @staticmethod
    def preprocess(image_bgr: np.ndarray) -> np.ndarray:
        """BGR image (as returned by cv2.imread) -> model input batch."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
        return rgb[np.newaxis, ...]

    def predict(self, image_bgr: np.ndarray) -> LivenessResult:
        """Classifies an image that is already loaded in memory (BGR order)."""
        batch = self.preprocess(image_bgr)
        score = float(np.squeeze(self.model.predict(batch, verbose=0)))
        return LivenessResult(score=score, is_live=score <= self.threshold)

    def predict_file(self, image_path: str):
        """Classifies an image file. Returns None if the file can't be read."""
        image = cv2.imread(image_path)
        if image is None:
            return None
        return self.predict(image)
