import os
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import timm


class XceptionModel:
    
    # Class names must match training
    CLASS_NAMES = ["Auto Rickshaws", "Bikes", "Cars", "Motorcycles", "Planes", "Ships", "Trains"]
    
    def __init__(self, model_dir: str, model_file: str = "best_model_finetuned_full.pt"):
        self.model_dir = model_dir
        self.model_file = model_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.inference_transform = None
        self.class_names = self.CLASS_NAMES
        
        print(f"[Xception] Using device: {self.device}")
        print(f"[Xception] Classes: {self.class_names}")
        self._load_model()
    
    def _load_model(self):
        try:
            model_path = os.path.join(self.model_dir, self.model_file)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Disable TorchDynamo (avoids CatchErrorsWrapper issues)
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.reset()
            
            # Load the model
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            num_classes = len(self.CLASS_NAMES)
            
            if isinstance(checkpoint, dict) and not hasattr(checkpoint, "forward"):
                # State dict: rebuild the model architecture used during training
                model = timm.create_model("xception", pretrained=False, num_classes=num_classes)
                in_features = model.get_classifier().in_features
                model.fc = nn.Sequential(
                    nn.Linear(in_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes),
                )
                
                state_dict = checkpoint
                if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                
                model.load_state_dict(state_dict)
            else:
                # Full model
                model = checkpoint
                if hasattr(model, "_orig_mod"):
                    model = model._orig_mod
            
            # Move model to device and set to evaluation mode
            self.model = model.to(self.device).eval()
            
            # Load preprocessing transforms
            data_config = timm.data.resolve_model_data_config(self.model)
            self.inference_transform = timm.data.create_transform(**data_config, is_training=False)
            
            print(f"[Xception] Model loaded successfully from {model_path}")
        
        except Exception as e:
            print(f"[Xception] Error loading model: {e}")
            raise
    
    def _preprocess_image(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        tensor = self.inference_transform(img).unsqueeze(0).to(self.device)
        return tensor
    
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
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=-1).cpu().numpy()[0]
            
            # Get predictions
            class_idx = int(np.argmax(probs))
            confidence = float(probs[class_idx])
            prob_dict = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
            
            return self.class_names[class_idx], confidence, prob_dict
        
        except Exception as e:
            print(f"[Xception] Error during prediction: {e}")
            raise
