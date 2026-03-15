import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np

class DiagramSegmentationModel:
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512"):
        """
        Initializes SegFormer. Using a lightweight model trained on ADE20K for quick demo purposes,
        which contains many object classes. For a production system for specific educational diagrams, 
        a custom fine-tuned model would be loaded here.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading segmentation model on {self.device}...")
        
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, image: Image.Image) -> tuple[np.ndarray, dict]:
        """
        Run inference on the image.
        Returns the segmentation mask and a dictionary mapping IDs to Labels.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Segformer outputs are shape (batch_size, num_classes, height, width)
        logits = outputs.logits
        
        # Resize logits to match the original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits, 
            size=image.size[::-1], # (height, width)
            mode="bilinear", 
            align_corners=False
        )
        
        # Get the predicted class for each pixel
        predictions = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # Ensure we have the labels config
        id2label = self.model.config.id2label
        
        return predictions, id2label
