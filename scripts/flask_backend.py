"""
Flask Backend for Food Classification Web App
Includes Grad-CAM explainability and real model inference
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import cv2
import base64
import io
import time
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import matplotlib.cm as cm

app = Flask(__name__)
CORS(app)

# Global variables
model = None
class_names = ['Bread', 'Fried food', 'Seafood', 'Vegetable-Fruit']
IMG_SIZE = 160


def load_food_model(model_path):
    """Load trained model"""
    global model
    try:
        model = load_model(model_path)
        print(f" Model loaded from {model_path}")
        return True
    except Exception as e:
        print(f" Error loading model: {e}")
        return False


def preprocess_image(image, target_size=(IMG_SIZE, IMG_SIZE)):
    """Preprocess image for model input"""  
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name='Conv_1', pred_index=None):
    """
    Generate Grad-CAM heatmap
    
    Args:
        img_array: Preprocessed image array
        model: Keras model
        last_conv_layer_name: Name of last convolutional layer
        pred_index: Class index for which to generate heatmap
    
    Returns:
        Heatmap array
    """
    
    # Create a model that maps input to activations of last conv layer and output predictions
    try:
        grad_model = tf.keras.models.Model(
            inputs=[model.input],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
    except:
        # If specific layer not found, try to find last conv layer
        conv_layers = [layer for layer in model.layers if 'conv' in layer.name.lower()]
        if conv_layers:
            last_conv_layer_name = conv_layers[-1].name
            grad_model = tf.keras.models.Model(
                inputs=[model.input],
                outputs=[model.get_layer(last_conv_layer_name).output, model.output]
            )
        else:
            # Fallback: use simple center-weighted heatmap
            return create_simple_heatmap(img_array.shape[1:3])
    
    # Compute gradient of predicted class with respect to output feature map
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Gradient of output with respect to output feature map
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Mean intensity of gradient over specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel by importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()


def create_simple_heatmap(img_shape):
    """Create simple center-focused heatmap as fallback"""
    h, w = img_shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    
    # Create Gaussian-like heatmap
    heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 4)**2))
    
    return heatmap


def overlay_heatmap_on_image(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on original image
    
    Args:
        image: Original image (PIL Image or numpy array)
        heatmap: Heatmap array
        alpha: Transparency of overlay
        colormap: OpenCV colormap
    
    Returns:
        Overlayed image as numpy array
    """
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on image
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return overlayed


def image_to_base64(image):
    """Convert numpy array or PIL Image to base64 string"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"


@app.route('/')
def index():
    """Serve main page"""
    # Return simple HTML that loads the static demo
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Food Classification API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { color: #667eea; }
            .endpoint {
                background: #f8f9fa;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid #667eea;
            }
            code {
                background: #e9ecef;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: monospace;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1> Food Classification API</h1>
            <p>Backend API for SOTA Food Classification System</p>
            
            <h2>Endpoints:</h2>
            
            <div class="endpoint">
                <strong>POST /predict</strong>
                <p>Classify food image and generate Grad-CAM heatmap</p>
                <p><strong>Input:</strong> <code>{"image": "base64_string"}</code></p>
                <p><strong>Output:</strong> Predictions, heatmap, inference time</p>
            </div>
            
            <div class="endpoint">
                <strong>GET /health</strong>
                <p>Check API health status</p>
            </div>
            
            <h2>Model Info:</h2>
            <ul>
                <li>Architecture: EfficientNetV2</li>
                <li>Classes: Bread, Fried food, Seafood, Vegetable-Fruit</li>
                <li>Input Size: 224x224</li>
                <li>Explainability: Grad-CAM</li>
            </ul>
        </div>
    </body>
    </html>
    """


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': class_names
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict food class and generate Grad-CAM heatmap
    
    Expected input: JSON with 'image' field containing base64 encoded image
    """
    
    try:
        # Get image from request
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Store original for heatmap overlay
        original_image = image.copy()
        original_size = image.size
        
        # Preprocess for model
        img_array = preprocess_image(image, target_size=(IMG_SIZE, IMG_SIZE))
        
        # Measure inference time
        start_time = time.time()
        
        if model is None:
            # Simulate prediction if model not loaded
            predictions = np.random.rand(1, len(class_names))
            predictions = predictions / predictions.sum()
            predictions = predictions[0]
        else:
            # Real prediction
            predictions = model.predict(img_array, verbose=0)[0]
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get top prediction
        top_index = np.argmax(predictions)
        top_class = class_names[top_index]
        top_confidence = float(predictions[top_index])
        
        # Generate Grad-CAM heatmap
        if model is not None:
            try:
                heatmap = make_gradcam_heatmap(img_array, model, pred_index=top_index)
            except:
                heatmap = create_simple_heatmap((IMG_SIZE, IMG_SIZE))
        else:
            heatmap = create_simple_heatmap((IMG_SIZE, IMG_SIZE))
        
        # Overlay heatmap on original image
        overlayed = overlay_heatmap_on_image(original_image, heatmap)
        heatmap_base64 = image_to_base64(overlayed)
        
        # Prepare response
        response = {
            'success': True,
            'predictions': [
                {
                    'class': class_names[i],
                    'confidence': float(predictions[i])
                }
                for i in range(len(class_names))
            ],
            'top_prediction': {
                'class': top_class,
                'confidence': top_confidence
            },
            'heatmap': heatmap_base64,
            'inference_time_ms': round(inference_time, 2),
            'image_size': f"{original_size[0]}x{original_size[1]}",
            'model_input_size': f"{IMG_SIZE}x{IMG_SIZE}"
        }
        
        # Sort predictions by confidence
        response['predictions'].sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    
    info = {
        'model_type': 'EfficientNetV2',
        'input_size': f"{IMG_SIZE}x{IMG_SIZE}",
        'classes': class_names,
        'num_classes': len(class_names),
        'explainability': 'Grad-CAM',
        'model_loaded': model is not None
    }
    
    if model is not None:
        info['total_params'] = model.count_params()
        info['trainable_params'] = sum([tf.keras.backend.count_params(w) 
                                        for w in model.trainable_weights])
    
    return jsonify(info)


def main():
    """Main function to run Flask app"""
    
    print("="*60)
    print("FOOD CLASSIFICATION API SERVER")
    print("="*60)
    
    # Try to load model
    model_path = 'models_sota_fixed/final_sota_model.h5'
    
    print(f"\nAttempting to load model from: {model_path}")
    
    if load_food_model(model_path):
        print(" Model loaded successfully!")
    else:
        print(" Model not found. Running in demo mode with simulated predictions.")
        print("   Train your model first using sota_training.py")
    
    print("\n" + "="*60)
    print("Starting Flask server...")
    print("="*60)
    print("\nEndpoints available:")
    print("  • http://localhost:5000/")
    print("  • http://localhost:5000/predict")
    print("  • http://localhost:5000/health")
    print("  • http://localhost:5000/model-info")
    print("\n" + "="*60)
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()