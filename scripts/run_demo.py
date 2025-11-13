"""
Demo Runner - Launch Food Classification Web Application
Combines Flask backend with HTML frontend
"""

import os
import sys
import webbrowser
import time
from threading import Thread
from flask import Flask, send_file

# Import backend
try:
    from flask_backend import app, load_food_model
except:
    print("Warning: Could not import flask_backend module")
    app = Flask(__name__)


def create_demo_html():
    """Create standalone demo HTML file"""
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Food Classification Demo</title>
    <style>
        body { 
            font-family: Arial; 
            max-width: 1200px; 
            margin: 50px auto; 
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        h1 { color: #667eea; text-align: center; }
        .status { 
            padding: 15px; 
            margin: 20px 0; 
            border-radius: 10px;
            text-align: center;
        }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .instructions {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .instructions h3 { color: #667eea; margin-bottom: 15px; }
        .instructions ol { margin-left: 20px; line-height: 2; }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1.1em;
            display: block;
            margin: 20px auto;
            text-decoration: none;
            text-align: center;
            width: fit-content;
        }
        .btn:hover { opacity: 0.9; }
    </style>
</head>
<body>
    <div class="container">
        <h1> Food Classification Demo</h1>
        
        <div class="status info" id="status">
            Checking server status...
        </div>
        
        <div class="instructions">
            <h3> How to Use:</h3>
            <ol>
                <li>Make sure Flask backend is running</li>
                <li>Click "Open Demo" button below</li>
                <li>Upload a food image or use camera</li>
                <li>View prediction results and Grad-CAM heatmap</li>
            </ol>
        </div>
        
        <a href="index.html" class="btn" id="demoBtn"> Open Demo</a>
        
        <div style="text-align: center; margin-top: 30px; color: #666;">
            <p><strong>Backend API:</strong> http://localhost:5000</p>
            <p><strong>Classes:</strong> Bread | Fried Food | Seafood | Vegetable-Fruit</p>
        </div>
    </div>
    
    <script>
        async function checkServer() {
            const statusDiv = document.getElementById('status');
            const demoBtn = document.getElementById('demoBtn');
            
            try {
                const response = await fetch('http://localhost:5000/health');
                const data = await response.json();
                
                if (data.status === 'healthy') {
                    statusDiv.className = 'status success';
                    statusDiv.innerHTML = ' Server is running! Model: ' + 
                        (data.model_loaded ? 'Loaded' : 'Demo Mode');
                    demoBtn.style.display = 'block';
                } else {
                    throw new Error('Server not healthy');
                }
            } catch (error) {
                statusDiv.className = 'status error';
                statusDiv.innerHTML = ' Server not running. Please start Flask backend first:<br>' +
                    '<code>python flask_backend.py</code>';
                demoBtn.style.display = 'none';
            }
        }
        
        checkServer();
        setInterval(checkServer, 5000);
    </script>
</body>
</html>'''
    
    with open('demo_launcher.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(" Demo launcher created: demo_launcher.html")


def open_browser():
    """Open browser after delay"""
    time.sleep(2)
    webbrowser.open('http://localhost:5000')


@app.route('/demo')
def demo_page():
    """Serve demo page"""
    try:
        return send_file('index.html')
    except:
        return """
        <html>
        <body style="font-family: Arial; text-align: center; padding: 50px;">
            <h1>Demo Page</h1>
            <p>Please create index.html in the project directory</p>
            <p>Or access the API directly at <a href="/">/</a></p>
        </body>
        </html>
        """


def print_banner():
    """Print startup banner"""
    
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     FOOD CLASSIFICATION WEB APPLICATION                    ║
    ║                                                               ║
    ║   Assignment 2 - Computer Vision                             ║
    ║   Astana IT University                                        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    
     FEATURES:
       • SOTA EfficientNetV2 Model
       • Grad-CAM Explainability
       • Real-time Classification
       • Camera Support
    
     ENDPOINTS:
       • Main Page:    http://localhost:5000
       • API Health:   http://localhost:5000/health
       • Prediction:   http://localhost:5000/predict
       • Model Info:   http://localhost:5000/model-info
       • Demo Page:    http://localhost:5000/demo
    
     CLASSES:
       1. Bread
       2. Fried Food
       3. Seafood
       4. Vegetable-Fruit
    
    ═══════════════════════════════════════════════════════════════
    """
    
    print(banner)


def check_dependencies():
    """Check if required packages are installed"""
    
    required_packages = {
        'flask': 'flask',
        'flask_cors': 'flask-cors',
        'tensorflow': 'tensorflow',
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib'
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("\nMissing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nInstall them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True


def main():
    """Main function to run demo"""
    
    print_banner()
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("\nPlease install missing packages first")
        return
    
    print("All dependencies installed\n")
    
    # Try to load model
    print("Loading model...")
    model_paths = [
        'models_sota_fixed/final_sota_model.h5',
        'models_sota_fixed/best_model.h5',
        'models/transfer_mobilenetv2.h5'
    ]
    
    model_loaded = False
    for path in model_paths:
        if os.path.exists(path):
            print(f"   Trying: {path}")
            if load_food_model(path):
                print(f"  Model loaded from {path}")
                model_loaded = True
                break
    
    if not model_loaded:
        print("   No trained model found")
        print("   Running in DEMO MODE with simulated predictions")
        print("   Train your model first using: python sota_training.py")
    
    print()
    
    # Create demo launcher
    create_demo_html()
    
    # Open browser in separate thread
    print("Starting web server...")
    Thread(target=open_browser, daemon=True).start()
    
    print("\nServer starting on http://localhost:5000")
    print("Press CTRL+C to stop\n")
    
    # Run Flask app
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")


if __name__ == '__main__':
    main()