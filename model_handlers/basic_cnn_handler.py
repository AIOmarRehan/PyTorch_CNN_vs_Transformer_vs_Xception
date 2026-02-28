import json
import os
from typing import List, Tuple, Dict
import torch
from PIL import Image
from torchvision import transforms
import numpy as np


class BasicCNNModel:
    
    class CNN(torch.nn.Module):
        def __init__(self, num_classes, dropout_rate=0.4):
            super(BasicCNNModel.CNN, self).__init__()
            self.dropout_rate = dropout_rate
            
            # Conv Block 1: 3 → 32 channels
            self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(32)
            self.relu1 = torch.nn.ReLU(inplace=True)
            self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Conv Block 2: 32 → 64 channels
            self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            self.bn2 = torch.nn.BatchNorm2d(64)
            self.relu2 = torch.nn.ReLU(inplace=True)
            self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Conv Block 3: 64 → 128 channels
            self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
            self.bn3 = torch.nn.BatchNorm2d(128)
            self.relu3 = torch.nn.ReLU(inplace=True)
            self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Conv Block 4: 128 → 256 channels
            self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
            self.bn4 = torch.nn.BatchNorm2d(256)
            self.relu4 = torch.nn.ReLU(inplace=True)
            self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Global Average Pooling (adaptive pooling to 1x1)
            self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            
            # Classifier head with Dropout
            self.dropout1 = torch.nn.Dropout(p=dropout_rate)
            self.fc1 = torch.nn.Linear(256, 512)
            self.fc1_relu = torch.nn.ReLU(inplace=True)
            
            self.dropout2 = torch.nn.Dropout(p=dropout_rate)
            self.fc2 = torch.nn.Linear(512, num_classes)
            
            # Initialize weights
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, torch.nn.BatchNorm2d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            # Block 1
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.maxpool1(x)
            
            # Block 2
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.maxpool2(x)
            
            # Block 3
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.maxpool3(x)
            
            # Block 4
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.maxpool4(x)
            
            # Global Average Pooling
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)  # Flatten
            
            # Classifier head
            x = self.dropout1(x)
            x = self.fc1(x)
            x = self.fc1_relu(x)
            
            x = self.dropout2(x)
            x = self.fc2(x)
            
            return x
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self.class_names = []
        self.metadata = None
        
        print(f"[BasicCNN] Using device: {self.device}")
        self._load_model()
    
    def _load_config(self) -> Dict:
        config_path = os.path.join(self.model_dir, "deployment_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r") as f:
            return json.load(f)
    
    def _load_metadata(self, metadata_path: str) -> Dict:
        with open(metadata_path, "r") as f:
            return json.load(f)
    
    def _build_transforms(self, mean: List[float], std: List[float]) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    def _load_model(self):
        try:
            config = self._load_config()
            metadata_path = os.path.join(self.model_dir, config["metadata"])
            state_dict_path = os.path.join(self.model_dir, config["model_state_dict"])
            
            self.metadata = self._load_metadata(metadata_path)
            
            # Load model
            self.model = self.CNN(num_classes=self.metadata["num_classes"], dropout_rate=0.4)
            state_dict = torch.load(state_dict_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            self.transform = self._build_transforms(
                self.metadata["normalization_mean"],
                self.metadata["normalization_std"]
            )
            
            # Load class names
            class_names_dict = self.metadata.get("class_names", {})
            self.class_names = [class_names_dict[str(i)] for i in range(len(class_names_dict))]
            
            print(f"[BasicCNN] Model loaded successfully. Classes: {self.class_names}")
        
        except Exception as e:
            print(f"[BasicCNN] Error loading model: {e}")
            raise
    
    def predict(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        if image is None:
            return "No image provided", 0.0, {}
        
        try:
            # Prepare image
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Get predictions
            class_idx = int(np.argmax(probs))
            confidence = float(probs[class_idx])
            prob_dict = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
            
            return self.class_names[class_idx], confidence, prob_dict
        
        except Exception as e:
            print(f"[BasicCNN] Error during prediction: {e}")
            raise
