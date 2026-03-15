import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import cv2

def create_color_palette(num_classes: int) -> np.ndarray:
    """
    Generate a distinct color for each class.
    """
    np.random.seed(42)  # For consistent colors
    palette = np.random.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
    return palette

def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Overlay a segmentation mask onto an image.
    Mask should be a 2D array of class indices.
    """
    img_array = np.array(image)
    
    # Get unique classes in the mask
    unique_classes = np.unique(mask)
    palette = create_color_palette(max(256, unique_classes.max() + 1))
    
    # Create an RGB image for the mask based on the palette
    mask_rgb = palette[mask]
    
    # Blend the original image and the mask
    overlayed = cv2.addWeighted(img_array, 1 - alpha, mask_rgb, alpha, 0)
    
    return Image.fromarray(overlayed)

def get_bounding_boxes(mask: np.ndarray):
    """
    Extract bounding boxes for each detected region in the mask.
    Returns a dictionary of class_id -> list of bounding boxes (x, y, w, h).
    """
    unique_classes = np.unique(mask)
    boxes_dict = {}
    
    for cls in unique_classes:
        if cls == 0: # Assuming 0 is background and we might want to skip it, but let's keep all for now or skip if desired.
            continue
            
        cls_mask = (mask == cls).astype(np.uint8)
        contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10: # Filter out very small noise
                boxes.append((x, y, w, h))
                
        if boxes:
            boxes_dict[cls] = boxes
            
    return boxes_dict

def draw_bounding_boxes(image: Image.Image, boxes_dict: dict, id2label: dict = None) -> Image.Image:
    """
    Draw bounding boxes on the image.
    """
    img_array = np.array(image)
    
    palette = create_color_palette(max(256, max(boxes_dict.keys()) + 1) if boxes_dict else 256)
    
    for cls, boxes in boxes_dict.items():
        color = tuple(int(x) for x in palette[cls])
        label = id2label[cls] if id2label and cls in id2label else f"Class {cls}"
        
        for (x, y, w, h) in boxes:
            cv2.rectangle(img_array, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_array, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    return Image.fromarray(img_array)
