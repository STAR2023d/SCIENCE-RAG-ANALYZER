"""Check that the vision model accepts an image and describes it.

Usage:
    uv run python test_vision.py "/path/to/any/image.png"
"""
import sys

from image_loader import caption_image, VISION_MODEL

if len(sys.argv) < 2:
    print('Usage: uv run python test_vision.py "/path/to/image.png"')
    sys.exit(1)

print("Model:", VISION_MODEL)
print(caption_image(sys.argv[1]))