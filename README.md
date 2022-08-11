# White Blood Cell Detection and Classification using Single Shot Multibox Detector and Deployment

---
This is a **[PyTorch](https://pytorch.org) Tutorial to Object Detection**.

This is blatanly "borrowed" from [sgrvinod's great tutorial](https://github.com/sgrvinod/Deep-Tutorials-for-PyTorch) 

Basic knowledge of PyTorch, convolutional neural networks is assumed.

---
# Set up
```
git clone https://github.com/tranduchuy682/SSD.git
pip install -r requirements.txt
```

---
# Get dataset
- The dataset can be downloaded or cloned from [***my other repo***](https://github.com/tranduchuy682/AllDatabase.git)
- Format
```
├── AllDatabase
|  └── BCCD Database
|  └── LISCDatabase
|  |    └── Ground Truth Segmentation
|  |    └── Main Dataset
|  |    └── More Dataset without Ground Truth/alll
|  └── RaabinDatabase
|       └── GrTh
|       └── TestA
|       └── TestB/Test-B
|       └── Train
├── train.py
├── utils.py
...
```
```
git clone https://github.com/tranduchuy682/AllDatabase.git
```

---
# Prepare the dataset for training and evaluate
```
python3 create_data_lists.py
```

---
# Training
```python
backbone == ["resnet18","mobilenetv3","vgg16"]
epoch = 150
```

```
python3 train.py resnet18 150
```

---
# Testing
```
python3 eval.py
```

# Inference
```
python3 detect.py
```

# Deploy
```
python3 app.py
```
