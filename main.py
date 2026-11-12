"""
Command-line demo for the face liveness detector.

    python main.py                         # every image in ./input
    python main.py --input photo.jpg       # a single image
    python main.py --input my_folder --threshold 0.1   # 0.1 is the default
"""

import argparse
import os
import sys

# Hide TensorFlow's informational start-up messages
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from liveness_detector import (  # noqa: E402
    BASE_DIR,
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHTS,
    SUPPORTED_FORMATS,
    LivenessDetector,
)


def find_images(path: str) -> list:
    """Returns the image at `path`, or every supported image inside it."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return [os.path.join(path, name) for name in sorted(os.listdir(path))
                if name.lower().endswith(SUPPORTED_FORMATS)]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face liveness (anti-spoofing) detection")
    parser.add_argument("--input", default=os.path.join(BASE_DIR, "input"),
                        help="image file or folder of images (default: ./input)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="path to the model weights (default: weights/liveness.hdf5)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="spoof scores above this are labeled Fake (default: 0.10)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    images = find_images(args.input)
    if not images:
        print(f"No images found at: {args.input}")
        return 1

    detector = LivenessDetector(args.weights, args.threshold)

    print(f"\n{'Image':<32}{'Result':<8}Spoof score")
    print("-" * 52)
    for image_path in images:
        name = os.path.basename(image_path)
        result = detector.predict_file(image_path)
        if result is None:
            print(f"{name:<32}{'-':<8}could not read image")
            continue
        print(f"{name:<32}{result.label:<8}{result.score:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
