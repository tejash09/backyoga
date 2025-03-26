import base64
import json
import os
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
from contextlib import asynccontextmanager

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

# Global variables to store models and configurations
movenet = None
pose_classifier = None
class_mapping = None
int_to_label = None

# ====== CONFIGURATION ======
MODEL_PATH = "yoga_pose_model (1).h5"
CLASS_MAPPING_PATH = "class_mapping.json"
CONFIDENCE_THRESHOLD = 0.9
KEYPOINT_THRESHOLD = 0.3

# TensorFlow optimization
tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Configure GPU memory growth to avoid OOM errors
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Found {len(physical_devices)} GPU(s), enabled memory growth")
    except Exception as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU found, using CPU")

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models and initialize resources
    import asyncio
    app.state.event_loop = asyncio.get_event_loop()
    
    # Load models in background
    load_models_task = asyncio.create_task(load_models_async())
    
    yield
    
    # Shutdown: clean up resources
    executor.shutdown(wait=False)


# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====== POSE PROCESSING FUNCTIONS ======
def get_center_point(landmarks, left_idx, right_idx):
    """Calculate the center point between two landmarks."""
    left = landmarks[left_idx]
    right = landmarks[right_idx]
    return (left + right) * 0.5


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculate pose size based on torso and overall dimensions."""
    hips_center = get_center_point(landmarks, 11, 12)
    shoulders_center = get_center_point(landmarks, 5, 6)
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
    
    return visibility_ratio >= 0.7


# Precompile and optimize functions with TensorFlow
def run_movenet(img):
    global movenet
    return movenet(img)


def process_image(frame_data: str):
    """Process image data in a separate thread."""
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
        
        # Resize image for MoveNet (using TensorFlow operations for speed)
        img = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 256, 256)
        img = tf.cast(img, dtype=tf.int32)
        
        return img
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None


def run_inference_task(img):
    """Run the actual inference in a separate thread."""
    global movenet, pose_classifier, int_to_label
    
    try:
        # Run MoveNet
        results = run_movenet(img)
        keypoints = results['output_0'].numpy()[0, 0, :, :]
        
        # Check if enough keypoints are visible
        if not check_keypoint_visibility(keypoints):
            return {
                "pose_name": "Not enough keypoints visible",
                "confidence": 0.0,
                "keypoints": keypoints.tolist(),
                "is_correct": False
            }
        
        # Extract keypoints for classification
        keypoints_xy = keypoints[:, :2]
        
        # Convert to embedding
        embedding = landmarks_to_embedding(keypoints_xy)
        embedding = np.expand_dims(embedding, axis=0)
        
        # Run classifier
        prediction = pose_classifier.predict(embedding, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[predicted_class])
        
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


async def run_inference(frame_data: str) -> Dict[str, Any]:
    """Run pose detection and classification on image data using thread pool."""
    global movenet, pose_classifier
    
    if movenet is None or pose_classifier is None:
        return {"error": "Models not loaded. Call /load_models first."}
    
    # Process image in thread pool
    img = await app.state.event_loop.run_in_executor(executor, process_image, frame_data)
    
    if img is None:
        return {"error": "Error processing image"}
    
    # Run inference in thread pool
    result = await app.state.event_loop.run_in_executor(executor, run_inference_task, img)
    
    return result


@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "models_loaded": movenet is not None and pose_classifier is not None,
        "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0
    }


async def load_models_async():
    """Async wrapper for load_models to be used with asyncio.create_task"""
    await load_models()


@app.get("/load_models")
async def load_models():
    global movenet, pose_classifier, class_mapping, int_to_label
    
    if movenet is not None and pose_classifier is not None:
        return {"status": "success", "message": "Models already loaded", "poses": list(int_to_label.values())}
    
    try:
        # Load MoveNet model
        print("Loading MoveNet model...")
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        movenet = module.signatures['serving_default']
        
        # Load yoga pose classifier
        if os.path.exists(MODEL_PATH):
            print(f"Loading yoga pose classifier from {MODEL_PATH}...")
            pose_classifier = keras.models.load_model(MODEL_PATH, compile=True)
            
            # Warm up the model with a dummy prediction
            dummy_input = np.zeros((1, 34))  # 17 keypoints * 2 (x,y)
            pose_classifier.predict(dummy_input, verbose=0)
        else:
            return {"status": "error", "message": f"Model file not found: {MODEL_PATH}"}
        
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
            int_to_label = {i: pose for i, pose in enumerate(poses)}
            class_mapping = {"int_to_label": int_to_label}
        
        return {"status": "success", "message": "Models loaded successfully", "poses": list(int_to_label.values())}
    
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return {"status": "error", "message": str(e)}


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            
            try:
                # Parse the JSON data
                json_data = json.loads(data)
                
                # Check if it's a command or an image
                if "command" in json_data:
                    if json_data["command"] == "load_models":
                        result = await load_models()
                        await websocket.send_json(result)
                    elif json_data["command"] == "health":
                        result = await health_check()
                        await websocket.send_json(result)
                    else:
                        await websocket.send_json({"error": "Unknown command"})
                
                # Process image data
                elif "image" in json_data:
                    result = await run_inference(json_data["image"])
                    await websocket.send_json(result)
                
                else:
                    await websocket.send_json({"error": "Invalid message format"})
            
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})
            except Exception as e:
                await websocket.send_json({"error": f"Error processing request: {str(e)}"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# For backwards compatibility, you can keep the REST endpoint too
@app.post("/detect_pose")
async def detect_pose(data: Dict[str, str], background_tasks: BackgroundTasks):
    if "image" not in data:
        return {"error": "No image data provided"}
    
    result = await run_inference(data["image"])
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1)
