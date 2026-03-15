import cv2
import numpy as np
from PIL import Image

def load_image(image_path: str) -> Image.Image:
    """
    Load an image from a given path.
    """
    return Image.open(image_path).convert("RGB")

def preprocess_image_for_model(image: Image.Image) -> Image.Image:
    """
    Standardize the image if required before passing to the AI model.
    For SegFormer, the processor usually handles resizing, but we can do basic checks here.
    """
    return image

def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    Resize image to avoid running out of memory during processing or API calls.
    """
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image
