

## 1 Code for training the heart segmentation model

- python 1.Heart_seg.py

## 2 Code for EAT extraction

- The pre-trained segmentation model is stored in the directory './unetmodel.pth'
- python 2.EAT-extract.py
- The prediction results are saved in the './pred/' directory (npy format file).

## code for object detection of YOLO-V5
- Refer to the code in the link [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

## code for feature extraction
- Refer to the code in the link [KD4COVID19/core/feature_analysis/feature_extractor.py](https://github.com/MIILab-MTU/KD4COVID19/blob/main/core/feature_analysis/feature_extractor.py)


## Statistical analysis for the severity of COVID-19 infection
- Refer to the code in the link [KD4COVID19/core/feature_analysis/feature_analyzer.py](https://github.com/MIILab-MTU/KD4COVID19/blob/main/core/feature_analysis/feature_analyzer.py)

