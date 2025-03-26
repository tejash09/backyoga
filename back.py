from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import json
import base64
import time
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ====== CONFIGURATION ======
MODEL_PATH = "yoga_pose_model (1).h5"  # Path to your trained model
CLASS_MAPPING_PATH = "class_mapping.json"  # Path to your class mapping file
CONFIDENCE_THRESHOLD = 0.9  # Minimum confidence to consider a pose correct
KEYPOINT_THRESHOLD = 0.3  # Minimum confidence for individual keypoints

# Global variables to store models
movenet = None
pose_classifier = None
class_mapping = None
int_to_label = None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "models_loaded": movenet is not None and pose_classifier is not None})

@app.route('/load_models', methods=['GET'])
def load_models():
    global movenet, pose_classifier, class_mapping, int_to_label
    
    try:
        # Load MoveNet model
        print("Loading MoveNet model...")
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        movenet = module.signatures['serving_default']
        
        # Load yoga pose classifier
        if os.path.exists(MODEL_PATH):
            print(f"Loading yoga pose classifier from {MODEL_PATH}...")
            pose_classifier = keras.models.load_model(MODEL_PATH)
        else:
            return jsonify({"status": "error", "message": f"Model file not found: {MODEL_PATH}"})
        
        # Load class mapping
        if os.path.exists(CLASS_MAPPING_PATH):
            print(f"Loading class mapping from {CLASS_MAPPING_PATH}...")
            with open(CLASS_MAPPING_PATH, "r") as f:
                class_mapping = json.load(f)
                int_to_label = {int(i): label for i, label in class_mapping["int_to_label"].items()}
                poses = [int_to_label[i] for i in range(len(int_to_label))]
                print("Loaded classes:", poses)
        else:
            # Fallback if class mapping file not found
            print("Class mapping file not found. Using default class names.")
            poses = [
                "Chair Pose", "Dolphin Plank Pose", "Downward-Facing Dog Pose", 
                "Fish Pose", "Goddess Pose", "Locust Pose", "Lord of the Dance Pose", 
                "Low Lunge Pose", "Seated Forward Bend Pose", "Side Plank Pose", 
                "Staff Pose", "Tree Pose (Vrikshasana)", "Warrior I (Virabhadrasana I)", 
                "Warrior II Pose", "Warrior III Pose", "Wide-Angle Seated Forward Bend Pose"
            ]
            # Create a simple mapping
            int_to_label = {i: pose for i, pose in enumerate(poses)}
            class_mapping = {"int_to_label": int_to_label}
        
        return jsonify({"status": "success", "message": "Models loaded successfully", "poses": list(int_to_label.values())})
    
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

# ====== POSE PROCESSING FUNCTIONS ======
def get_center_point(landmarks, left_idx, right_idx):
    """Calculate the center point between two landmarks."""
    left = landmarks[left_idx]
    right = landmarks[right_idx]
    return (left + right) * 0.5

def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculate pose size based on torso and overall dimensions."""
    hips_center = get_center_point(landmarks, 11, 12)  # LEFT_HIP=11, RIGHT_HIP=12
    shoulders_center = get_center_point(landmarks, 5, 6)  # LEFT_SHOULDER=5, RIGHT_SHOULDER=6
    torso_size = np.linalg.norm(shoulders_center - hips_center)
    pose_center = hips_center
    dists = np.linalg.norm(landmarks - pose_center, axis=1)
    max_dist = np.max(dists)
    return max(torso_size * torso_size_multiplier, max_dist)

def normalize_pose_landmarks(landmarks):
    """Normalize landmarks to be centered and scaled."""
    pose_center = get_center_point(landmarks, 11, 12)
    landmarks = landmarks - pose_center
    pose_size = get_pose_size(landmarks)
    return landmarks / pose_size

def landmarks_to_embedding(landmarks):
    """Convert normalized landmarks to embedding vector."""
    norm_landmarks = normalize_pose_landmarks(landmarks)
    return norm_landmarks.flatten()

def check_keypoint_visibility(keypoints, threshold=KEYPOINT_THRESHOLD):
    """Check if enough keypoints are visible with sufficient confidence."""
    visible_keypoints = sum(1 for kp in keypoints if kp[2] > threshold)
    total_keypoints = len(keypoints)
    visibility_ratio = visible_keypoints / total_keypoints
    
    return visibility_ratio >= 0.7  # Require at least 70% of keypoints to be visible

def run_inference(frame_data):
    """Run pose detection and classification on image data."""
    global movenet, pose_classifier, int_to_label
    
    if movenet is None or pose_classifier is None:
        return {"error": "Models not loaded. Call /load_models first."}
    
    # Decode base64 image
    try:
        # Remove data URL prefix if present
        if "base64," in frame_data:
            frame_data = frame_data.split("base64,")[1]
            
        # Decode base64 to bytes
        img_bytes = base64.b64decode(frame_data)
        
        # Convert to numpy array
        np_arr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode to image
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB (MoveNet expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return {"error": f"Error decoding image: {str(e)}"}
    
    try:
        # Resize image and run pose estimation
        img = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 256, 256)
        img = tf.cast(img, dtype=tf.int32)
        
        # Run MoveNet
        results = movenet(img)
        keypoints = results['output_0'].numpy()[0, 0, :, :]  # Shape [17, 3]
        
        # Check if enough keypoints are visible
        if not check_keypoint_visibility(keypoints):
            return {
                "pose_name": "Not enough keypoints visible",
                "confidence": 0.0,
                "keypoints": keypoints.tolist(),
                "is_correct": False
            }
        
        # Extract keypoints for classification (just the x,y coordinates)
        keypoints_xy = keypoints[:, :2]
        
        # Convert to embedding
        embedding = landmarks_to_embedding(keypoints_xy)
        embedding = np.expand_dims(embedding, axis=0)  # Add batch dimension
        
        # Run classifier
        prediction = pose_classifier.predict(embedding, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[predicted_class])  # Convert to Python float for JSON serialization
        
        # Get pose name
        pose_name = int_to_label.get(predicted_class, f"Unknown pose ({predicted_class})")
        
        # Determine if pose is correct based on confidence
        is_correct = confidence >= CONFIDENCE_THRESHOLD
        
        return {
            "pose_name": pose_name,
            "confidence": confidence,
            "keypoints": keypoints.tolist(),
            "is_correct": is_correct
        }
    
    except Exception as e:
        print(f"Error in inference: {str(e)}")
        return {"error": str(e)}

@app.route('/detect_pose', methods=['POST'])
def detect_pose():
    if not request.json or 'image' not in request.json:
        return jsonify({"error": "No image data provided"}), 400
    
    frame_data = request.json['image']
    result = run_inference(frame_data)
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
