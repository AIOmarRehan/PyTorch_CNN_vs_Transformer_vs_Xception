import os
from typing import Tuple, Dict
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor


class HuggingFaceModel:
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.class_names = []
        
        print(f"[HuggingFace] Using device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        try:
            # Load model and processor
            self.model = AutoModelForImageClassification.from_pretrained(self.model_dir)
            self.processor = AutoImageProcessor.from_pretrained(self.model_dir)
            
            # Move to device
            self.model.to(self.device).eval()
            
            # Get class names from model config
            self.class_names = list(self.model.config.id2label.values())
            
            print(f"[HuggingFace] Model loaded successfully. Classes: {self.class_names}")
        
        except Exception as e:
            print(f"[HuggingFace] Error loading model: {e}")
            raise
    
    def _preprocess_image(self, img: Image.Image) -> Dict:
        inputs = self.processor(images=img, return_tensors='pt')
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def predict(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:

        if image is None:
            return "No image provided", 0.0, {}
        
        try:
            # Ensure image is PIL Image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            
            # Preprocess image
            inputs = self._preprocess_image(image)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # Get predictions
            class_idx = int(np.argmax(probs))
            confidence = float(probs[class_idx])
            prob_dict = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
            
            return self.class_names[class_idx], confidence, prob_dict
        
        except Exception as e:
            print(f"[HuggingFace] Error during prediction: {e}")
            raise
