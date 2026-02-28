import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict
from PIL import Image

import gradio as gr
import torch

# Import model handlers
from model_handlers.basic_cnn_handler import BasicCNNModel
from model_handlers.hugging_face_handler import HuggingFaceModel
from model_handlers.xception_handler import XceptionModel


# Global Configuration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_1_DIR = os.path.join(MODELS_DIR, "basic_cnn")
MODEL_2_DIR = os.path.join(MODELS_DIR, "hugging_face")
MODEL_3_DIR = os.path.join(MODELS_DIR, "xception")

# Model instances (loaded at startup)
basic_cnn_model = None
hugging_face_model = None
xception_model = None

MODELS_INFO = {
    "Model 1: Basic CNN": {
        "description": "Custom CNN architecture with 4 Conv blocks and BatchNorm",
        "path": MODEL_1_DIR,
        "handler_class": BasicCNNModel
    },
    "Model 2: Hugging Face Transformers": {
        "description": "Pre-trained transformer-based model from Hugging Face",
        "path": MODEL_2_DIR,
        "handler_class": HuggingFaceModel
    },
    "Model 3: Xception CNN": {
        "description": "Fine-tuned Xception architecture using timm library",
        "path": MODEL_3_DIR,
        "handler_class": XceptionModel
    }
}


# Model Loading

def load_models():
    """Load all three models at startup"""
    global basic_cnn_model, hugging_face_model, xception_model
    
    print("\n" + "="*60)
    print("Loading Models...")
    print("="*60)
    
    try:
        print("\n[1/3] Loading Basic CNN Model...")
        basic_cnn_model = BasicCNNModel(MODEL_1_DIR)
        print("Basic CNN Model loaded successfully")
    except Exception as e:
        print(f"Failed to load Basic CNN Model: {e}")
        basic_cnn_model = None
    
    try:
        print("\n[2/3] Loading Hugging Face Model...")
        hugging_face_model = HuggingFaceModel(MODEL_2_DIR)
        print("Hugging Face Model loaded successfully")
    except Exception as e:
        print(f"Failed to load Hugging Face Model: {e}")
        hugging_face_model = None
    
    try:
        print("\n[3/3] Loading Xception Model...")
        xception_model = XceptionModel(MODEL_3_DIR)
        print("Xception Model loaded successfully")
    except Exception as e:
        print(f"Failed to load Xception Model: {e}")
        xception_model = None
    
    print("\n" + "="*60)
    print("Model Loading Complete!")
    print("="*60 + "\n")


# Prediction Functions

def predict_with_model_1(image: Image.Image) -> Tuple[str, float, Dict]:
    """Predict with Basic CNN Model"""
    if basic_cnn_model is None:
        return "Model 1: Error", 0.0, {}
    try:
        label, confidence, prob_dict = basic_cnn_model.predict(image)
        return label, confidence, prob_dict
    except Exception as e:
        print(f"Error in Model 1 prediction: {e}")
        return "Error", 0.0, {}


def predict_with_model_2(image: Image.Image) -> Tuple[str, float, Dict]:
    """Predict with Hugging Face Model"""
    if hugging_face_model is None:
        return "Model 2: Error", 0.0, {}
    try:
        label, confidence, prob_dict = hugging_face_model.predict(image)
        return label, confidence, prob_dict
    except Exception as e:
        print(f"Error in Model 2 prediction: {e}")
        return "Error", 0.0, {}


def predict_with_model_3(image: Image.Image) -> Tuple[str, float, Dict]:
    """Predict with Xception Model"""
    if xception_model is None:
        return "Model 3: Error", 0.0, {}
    try:
        label, confidence, prob_dict = xception_model.predict(image)
        return label, confidence, prob_dict
    except Exception as e:
        print(f"Error in Model 3 prediction: {e}")
        return "Error", 0.0, {}


def predict_all_models(image: Image.Image):
    if image is None:
        empty_result = {"Model": "N/A", "Prediction": "No image", "Confidence": 0.0}
        empty_probs = {}
        empty_consensus = "<p>Please upload an image to see results</p>"
        return empty_result, empty_result, empty_result, "Please upload an image", empty_probs, empty_probs, empty_probs, empty_consensus
    
    print("\n" + "="*60)
    print("Running Predictions with All Models...")
    print("="*60)
    
    # Run predictions in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_1 = executor.submit(predict_with_model_1, image)
        future_2 = executor.submit(predict_with_model_2, image)
        future_3 = executor.submit(predict_with_model_3, image)
        
        # Wait for all predictions to complete
        result_1_label, result_1_conf, result_1_probs = future_1.result()
        result_2_label, result_2_conf, result_2_probs = future_2.result()
        result_3_label, result_3_conf, result_3_probs = future_3.result()
    
    # Format results for display
    result_1 = {
        "Model": "Basic CNN",
        "Prediction": result_1_label,
        "Confidence": f"{result_1_conf * 100:.2f}%"
    }
    
    result_2 = {
        "Model": "Hugging Face",
        "Prediction": result_2_label,
        "Confidence": f"{result_2_conf * 100:.2f}%"
    }
    
    result_3 = {
        "Model": "Xception",
        "Prediction": result_3_label,
        "Confidence": f"{result_3_conf * 100:.2f}%"
    }
    
    # Check if all models agree
    all_agree = result_1_label == result_2_label == result_3_label
    
    # Create comparison text with HTML styling
    if all_agree:
        consensus_html = f"""
        <div style="background-color: #d4edda; border: 2px solid #28a745; border-radius: 8px; padding: 20px; text-align: center;">
            <h3 style="color: #155724; margin: 0; font-size: 24px;">All Models Agree!</h3>
            <p style="color: #155724; margin: 10px 0 0 0; font-size: 18px; font-weight: bold;">{result_1_label}</p>
        </div>
        """
    else:
        consensus_html = f"""
        <div style="background-color: #f8d7da; border: 2px solid #dc3545; border-radius: 8px; padding: 20px; text-align: center;">
            <h3 style="color: #721c24; margin: 0; font-size: 24px;">Models Disagree</h3>
            <p style="color: #721c24; margin: 10px 0 0 0; font-size: 16px;">Check predictions below for details</p>
        </div>
        """
    
    comparison_text = f"""
    ## Comparison Results
    
    **Model 1 (Basic CNN):** {result_1_label} ({result_1_conf * 100:.2f}%)
    
    **Model 2 (Hugging Face):** {result_2_label} ({result_2_conf * 100:.2f}%)
    
    **Model 3 (Xception):** {result_3_label} ({result_3_conf * 100:.2f}%)
    """
    
    print(f"Prediction 1: {result_1_label} ({result_1_conf * 100:.2f}%)")
    print(f"Prediction 2: {result_2_label} ({result_2_conf * 100:.2f}%)")
    print(f"Prediction 3: {result_3_label} ({result_3_conf * 100:.2f}%)")
    print(f"Consensus: {'All agree!' if all_agree else 'Disagreement detected'}")
    print("="*60 + "\n")
    
    return result_1, result_2, result_3, comparison_text, result_1_probs, result_2_probs, result_3_probs, consensus_html


# Gradio Interface

def build_interface() -> gr.Blocks:
    """Build the Gradio interface"""
    
    with gr.Blocks(
        title="PyTorch Unified Model Comparison",
        theme=gr.themes.Soft()
    ) as demo:
        
        # Header
        gr.Markdown("""
        # PyTorch Unified Model Comparison
        
        Upload an image and compare predictions from three different PyTorch models **simultaneously**.
        
        This tool helps you understand how different architectures (Basic CNN, Transformers, Xception)
        classify the same image and identify where they agree or disagree.
        """)
        
        # Model Information
        with gr.Accordion("Model Information", open=False):
            gr.Markdown(f"""
            ### Model 1: Basic CNN
            - **Description:** {MODELS_INFO['Model 1: Basic CNN']['description']}
            - **Architecture:** 4 Conv blocks + BatchNorm + Global Avg Pooling
            - **Input Size:** 224×224
            
            ### Model 2: Hugging Face Transformers
            - **Description:** {MODELS_INFO['Model 2: Hugging Face Transformers']['description']}
            - **Framework:** transformers library
            
            ### Model 3: Xception CNN
            - **Description:** {MODELS_INFO['Model 3: Xception CNN']['description']}
            - **Architecture:** Fine-tuned Xception with timm
            """)
        
        # Input Section
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    sources=["upload", "webcam"]
                )
                predict_btn = gr.Button("Predict with All Models", variant="primary", size="lg")
        
        # Output Section
        gr.Markdown("## Results")
        
        with gr.Row():
            with gr.Column():
                result_1_box = gr.JSON(label="Model 1: Basic CNN")
            with gr.Column():
                result_2_box = gr.JSON(label="Model 2: Hugging Face")
            with gr.Column():
                result_3_box = gr.JSON(label="Model 3: Xception")
        
        # Comparison Section
        comparison_output = gr.Markdown(label="Comparison Summary")
        
        # Consensus Indicator (HTML for colored styling)
        consensus_output = gr.HTML(value="<p></p>")
        
        # Class Probabilities Section
        gr.Markdown("## Class Probabilities")
        
        with gr.Row():
            with gr.Column():
                probs_1 = gr.Label(label="Model 1 Probabilities")
            with gr.Column():
                probs_2 = gr.Label(label="Model 2 Probabilities")
            with gr.Column():
                probs_3 = gr.Label(label="Model 3 Probabilities")
        
        # Connect button click
        predict_btn.click(
            fn=predict_all_models,
            inputs=image_input,
            outputs=[result_1_box, result_2_box, result_3_box, comparison_output, probs_1, probs_2, probs_3, consensus_output]
        )
        
        # Also trigger on image upload
        image_input.change(
            fn=predict_all_models,
            inputs=image_input,
            outputs=[result_1_box, result_2_box, result_3_box, comparison_output, probs_1, probs_2, probs_3, consensus_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        
        **Available Classes:** Auto Rickshaws | Bikes | Cars | Motorcycles | Planes | Ships | Trains
        
        This unified application allows real-time comparison of three different deep learning models
        to understand their individual strengths and weaknesses.
        """)
    
    return demo


# Main Entry Point

if __name__ == "__main__":
    # Load all models at startup
    load_models()
    
    # Build and launch Gradio interface
    demo = build_interface()
    
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    
    print(f"\nLaunching Gradio Interface on {server_name}:{server_port}")
    print("Open your browser and navigate to http://localhost:7860\n")
    
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_error=True
    )
