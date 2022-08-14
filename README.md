# White Blood Cell Detection and Classification using Single Shot Multibox Detector and Deployment

---
This is a **[PyTorch](https://pytorch.org) Tutorial to Object Detection**.

This is blatanly "borrowed" from [sgrvinod's great tutorial](https://github.com/sgrvinod/Deep-Tutorials-for-PyTorch) 

Basic knowledge of PyTorch, convolutional neural networks is assumed.

---
# Set up
- Run command
```
git clone https://github.com/tranduchuy682/SSD.git
pip install -r requirements.txt
```
- Or run [.sh file](run.sh)

---
# Get dataset
- The dataset can be downloaded or cloned [***here***](https://github.com/tranduchuy682/AllDatabase.git)
- Format
```
├── AllDatabase
|  ├── BCCD Database
|  ├── LISCDatabase
|  |    └── Ground Truth Segmentation
|  |    └── Main Dataset
|  |    └── More Dataset without Ground Truth/alll
|  ├── RaabinDatabase
|  |    └── GrTh
|  |    └── TestA
|  |    └── TestB/Test-B
|  |    └── Train
|  ├── bbox.csv
|  ├── test_bbox.csv
|  └── train_bbox.csv
├── train.py
├── utils.py
...
```
- Run
```
cd ssd
git clone https://github.com/tranduchuy682/AllDatabase.git
```

---
# Prepare the dataset for training and evaluate
- Run command
```
python3 create_data_lists.py
```

---
# Training
- Backbones:
There are 3 backbone have been used
* VGG16 backbone
    <p align="center">
    <img src="img\VGG16.png">
    </p>
* Resnet18 backbone
- We have modified our resnet18 backbone to get resnet18* - more suitable with our dataset
    <p align="center">
    <img src="img\resnet18.png">
    </p>
* MobileNetV3 backbone
    <p align="center">
    <img src="img\MobilenetV3.png">
    </p>
- Config
```python
backbone == ["resnet18","mobilenetv3","vgg16"]
epoch = 150 or any int value
```
- Run command
```
python3 train.py resnet18 150
```

---
# Testing
- Run command
```
python3 eval.py
```

# Inference
- Run command
```
python3 detect.py
```

# Deploy
- Run command
```
python3 app.py
```
- Demo
    <p align="center">
    <img src="img\Demo.png">
    </p>
