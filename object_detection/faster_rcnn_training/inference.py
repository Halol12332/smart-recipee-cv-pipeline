import os
import json
import random
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- 1. SETUP & CONFIGURATION ---
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_path = "faster_rcnn_fridge_model.pth"
base_dir = "myproject-7"
valid_dir = os.path.join(base_dir, "valid")
valid_ann = os.path.join(valid_dir, "_annotations.coco.json")

# Confidence threshold (only draw boxes if the model is > 50% sure)
CONFIDENCE_THRESHOLD = 0.5 

# --- 2. AUTOMATICALLY EXTRACT CLASS NAMES ---
with open(valid_ann) as f:
    coco_data = json.load(f)

# Create a dictionary to map ID numbers to actual ingredient names (e.g., 1: "egg")
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
NUM_CLASSES = max(categories.keys()) + 1 
print(f"Loaded {len(categories)} ingredient classes.")

# --- 3. MODEL DEFINITION ---
def get_model(num_classes):
    # weights=None because we are loading our own trained weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Initialize and load weights
print("Loading trained model...")
model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval() # CRITICAL: Sets model to evaluation/testing mode

# --- 4. PICK A RANDOM UNSEEN IMAGE ---
# Get all image files from the valid folder
image_files = [f for f in os.listdir(valid_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
random_image_name = random.choice(image_files)
image_path = os.path.join(valid_dir, random_image_name)

print(f"Running inference on: {random_image_name}")

# --- 5. INFERENCE & DRAWING ---
# Load image with OpenCV
img_cv2 = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

# Convert to PyTorch tensor (scales pixels between 0 and 1)
img_tensor = torchvision.transforms.functional.to_tensor(img_rgb).unsqueeze(0).to(device)

# Pass image through the model
with torch.no_grad():
    prediction = model(img_tensor)[0]

# Extract boxes, labels, and scores
boxes = prediction['boxes'].cpu().numpy()
labels = prediction['labels'].cpu().numpy()
scores = prediction['scores'].cpu().numpy()

# Draw the results
for i, box in enumerate(boxes):
    if scores[i] >= CONFIDENCE_THRESHOLD:
        x_min, y_min, x_max, y_max = map(int, box)
        class_id = labels[i]
        class_name = categories.get(class_id, "Unknown")
        score = scores[i]

        # Draw bounding box (Green color, thickness 2)
        cv2.rectangle(img_cv2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Add label text
        label_text = f"{class_name}: {score:.2f}"
        cv2.putText(img_cv2, label_text, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the output image
output_filename = "inference_result.jpg"
cv2.imwrite(output_filename, img_cv2)
print(f"Success! Output saved as '{output_filename}' in your current folder.")
