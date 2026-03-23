import os
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# --- 1. REUSE YOUR DATASET CLASS ---
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
        target = {"boxes": boxes, "labels": labels}
        img = torchvision.transforms.functional.to_tensor(img)
        return img, target

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- 2. EVALUATION PIPELINE ---
if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Initializing Evaluation...")

    # Load Validation Data
    base_dir = "myproject-7"
    valid_dir = os.path.join(base_dir, "valid")
    valid_ann = os.path.join(valid_dir, "_annotations.coco.json")

    with open(valid_ann) as f:
        coco_data = json.load(f)
        
    NUM_CLASSES = max([cat['id'] for cat in coco_data['categories']]) + 1
    category_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}

    valid_dataset = FridgeDataset(valid_dir, valid_ann)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load("faster_rcnn_fridge_best.pth", map_location=device))
    model.to(device)
    model.eval() 

    # THE FIX: Initialize three separate calculators to force it to output all per-class data
    metric_main = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
    metric_50 = MeanAveragePrecision(box_format='xyxy', class_metrics=True, iou_thresholds=[0.5])
    metric_75 = MeanAveragePrecision(box_format='xyxy', class_metrics=True, iou_thresholds=[0.75])

    print("Running validation images through the model...")
    with torch.no_grad():
        for images, targets in tqdm(valid_loader):
            images = list(image.to(device) for image in images)
            preds = model(images)
            
            metric_preds = []
            metric_targets = []
            
            for i in range(len(preds)):
                metric_preds.append(dict(
                    boxes=preds[i]['boxes'].cpu(),
                    scores=preds[i]['scores'].cpu(),
                    labels=preds[i]['labels'].cpu(),
                ))
                metric_targets.append(dict(
                    boxes=targets[i]['boxes'],
                    labels=targets[i]['labels']
                ))
                
            metric_main.update(metric_preds, metric_targets)
            metric_50.update(metric_preds, metric_targets)
            metric_75.update(metric_preds, metric_targets)

    print("\nCalculating Final Scores...")
    res_main = metric_main.compute()
    res_50 = metric_50.compute()
    res_75 = metric_75.compute()
    
    print("\n" + "="*65)
    print("                 FASTER R-CNN OVERALL RESULTS")
    print("="*65)
    print(f"Overall mAP50-95 (Strict Accuracy):   {res_main['map'].item():.4f}")
    print(f"Overall mAP@50 (Standard Accuracy):   {res_main['map_50'].item():.4f}")
    print(f"Overall mAP@75 (Rigorous Accuracy):   {res_main['map_75'].item():.4f}")
    
    print("\n" + "="*65)
    print("              FASTER R-CNN CLASS-BY-CLASS RESULTS")
    print("="*65)
    print(f"{'Class Name':>16} | {'mAP@50':>10} | {'mAP@75':>10} | {'mAP50-95':>10}")
    print("-" * 65)
    
    # Extract the individual class arrays
    classes_eval = res_main['classes'].tolist()
    map_pc = res_main['map_per_class'].tolist()
    map_50_pc = res_50['map_per_class'].tolist()
    map_75_pc = res_75['map_per_class'].tolist()
    
    for idx, class_id in enumerate(classes_eval):
        class_name = category_dict.get(class_id, f"Unknown ({class_id})")
        m50 = map_50_pc[idx]
        m75 = map_75_pc[idx]
        m50_95 = map_pc[idx]
        print(f"{class_name:>16} | {m50:10.4f} | {m75:10.4f} | {m50_95:10.4f}")
    print("="*65)
