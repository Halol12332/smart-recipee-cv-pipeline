# Smart Recipee: Computer Vision Pipeline 🥦🍳

**Author:** Jaya Hakim Prajna  
**Date:** 23 March 2026  

This repository contains the Machine Learning and Computer Vision training pipelines for **Smart Recipee**, an AI-powered recipe recommendation web application. This project forms the Computer Vision pillar of my Final Year Project (FYP), designed to automatically detect ingredients inside a refrigerator and assess their freshness to provide zero-waste recipe recommendations.

---

## 🏗️ Project Architecture

This repository focuses exclusively on the model training, evaluation, and dataset preparation pipelines. The system is split into two primary tasks:

1. **Ingredient Detection (Object Detection):** Identifying and localizing multiple ingredients in highly cluttered, real-world refrigerator environments.
2. **Freshness Classification (Image Classification):** Evaluating highly perishable items (like tomatoes and leafy greens) to determine if they are fresh or rotten.

---

## 📂 Repository Structure

```text
FYP_PROJECT/
├── freshness_classification/       # Freshness evaluation pipeline
│   ├── dataset/                    # Curated Kaggle dataset (Fresh vs Rotten)
│   ├── runs/                       # Training logs and weights
│   ├── evaluate_mobilenet.py       # MobileNetV2 evaluation script
│   ├── evaluate_yolo.py            # YOLOv8n-cls evaluation script
│   ├── prepare_data.py             # Dataset formatting and splitting
│   ├── train_mobilenet.py          # MobileNetV2 training script
│   └── train_yolov8ncls.py         # YOLOv8n-cls training script
│
├── object_detection/               # Ingredient detection pipeline
│   ├── faster_rcnn_training/       # Faster R-CNN implementation
│   │   ├── myproject-7/            # COCO-formatted dataset
│   │   ├── evaluate.py             # mAP calculation script
│   │   ├── inference.py            # Bounding box visualizer
│   │   └── train.py                # PyTorch training loop
│   ├── rt_detr_training/           # RT-DETR (Vision Transformer) implementation
│   │   ├── evaluate.py             
│   │   └── train.rtdetr.py         
│   └── yolov8_training/            # YOLOv8 CNN implementation
│       ├── evaluate.py             
│       └── train.py                
│
├── README.md                       # Project documentation
└── requirements.txt                # Python environment dependencies
```
You have to download the dataset first. Link provided: ...
