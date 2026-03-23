import os
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# --- 1. CUSTOM COCO DATASET CLASS ---
class FridgeDataset(Dataset):
    def __init__(self, root_dir, annotation_file):
        self.root_dir = root_dir
        with open(annotation_file) as f:
            data = json.load(f)
            
        self.images = data['images']
        self.annotations = data['annotations']
        
        self.img_to_anns = {img['id']: [] for img in self.images}
        for ann in self.annotations:
            self.img_to_anns[ann['image_id']].append(ann)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        
        anns = self.img_to_anns[img_info['id']]
        boxes = []
        labels = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h]) 
            labels.append(ann['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_info['id']])}
        img = torchvision.transforms.functional.to_tensor(img)
        
        return img, target

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    return tuple(zip(*batch))

# --- 2. MODEL DEFINITION ---
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- 3. TRAINING LOOP (100 EPOCHS / 4GB VRAM SAFE) ---
if __name__ == "__main__":
    torch.cuda.empty_cache() 
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Hardware Check: Training on {device}")
    
    if torch.cuda.is_available():
        # Capping VRAM at 80% to keep the system stable during the long run
        torch.cuda.set_per_process_memory_fraction(0.8, 0)
        print("VRAM usage successfully capped at 80%!")

    # Setup Paths
    base_dir = "myproject-7"
    train_dir = os.path.join(base_dir, "train")
    train_ann = os.path.join(train_dir, "_annotations.coco.json")

    # Dynamically calculate the number of classes
    with open(train_ann) as f:
        coco_data = json.load(f)
    max_cat_id = max([cat['id'] for cat in coco_data['categories']])
    NUM_CLASSES = max_cat_id + 1
    print(f"Dataset Loaded: Detected {max_cat_id} ingredients + 1 background class.")

    # Initialize Dataset and DataLoader
    train_dataset = FridgeDataset(train_dir, train_ann)
    
    # Matching YOLOv8 settings where architecturally possible
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2,      # Kept at 2 to prevent Out-Of-Memory crashes
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,     # Matching your YOLOv8 config
        pin_memory=True 
    )

    model = get_model(NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0005)
    scaler = torch.amp.GradScaler('cuda')

    # Setting target to 100 Epochs
    num_epochs = 100
    best_loss = float('inf') # Set starting best loss to infinity
    
    print(f"\nStarting RTX-Optimized Training for {num_epochs} Epochs...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        loop = tqdm(train_loader, leave=True)
        
        for images, targets in loop:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += losses.item()
            
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=losses.item())
            
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"-> Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

        # --- CHECKPOINTING SYSTEM ---
        # Save a backup of the very last epoch no matter what
        torch.save(model.state_dict(), "faster_rcnn_fridge_latest.pth")
        
        # Save a separate file for the best performing epoch
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), "faster_rcnn_fridge_best.pth")
            print(f"*** New best loss achieved! Saved to 'faster_rcnn_fridge_best.pth' ***")

    print("\nTraining complete! Your best weights are saved and ready for evaluation.")
