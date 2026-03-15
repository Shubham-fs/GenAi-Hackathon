import numpy as np
from collections import Counter

def extract_detected_regions(mask: np.ndarray, id2label: dict) -> list[str]:
    """
    Extracts a list of unique labels detected in the segmentation mask.
    This gives the GenAI model a structured understanding of 'what is in the scene'.
    """
    # Count pixels per class
    unique_classes, counts = np.unique(mask, return_counts=True)
    
    detected_objects = []
    
    # Threshold for noise (e.g. at least 100 pixels to be considered a viable region)
    min_pixels_threshold = 100 
    
    for cls, count in zip(unique_classes, counts):
        if count > min_pixels_threshold and cls in id2label:
            label = id2label[cls]
            # often class 0 or 'wall', 'building', 'background' might be ignored in a diagram context
            if label not in ['background']:
                detected_objects.append(label)
                
    return detected_objects

def format_regions_for_prompt(detected_objects: list[str]) -> str:
    """
    Format the list of detected objects string array into a clean prompt format.
    """
    if not detected_objects:
        return "No specific components detected. Use general reasoning over the image."
        
    formatted = "Detected Components:\n"
    for obj in detected_objects:
        formatted += f"- {obj}\n"
    return formatted
