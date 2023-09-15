# License Plate Detection with YOLOv8

## Project Overview

This project aims to design and implement a real-time license plate detection system capable of accurately detecting and localizing license plates in images. The system is robust and capable of handling various environmental conditions, including different lighting conditions, vehicle orientations, and background clutter. The goal is to provide reliable results for further processing or use in applications such as traffic monitoring, parking management, or law enforcement.

## Dataset

The dataset used for this project consists of 367 images, each paired with a label file containing bounding box coordinates for the license plates. The dataset is available [here]([link_to_dataset](https://drive.google.com/drive/folders/16i5y-OXPfet1w1g9rUlK65moHvTx81Au?usp=sharing)), and it serves as the foundation for training and evaluating the license plate detection model.
dataset/
    
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   ├── ...
    │   └── image367.jpg
    └── labels/
        ├── image1.txt
        ├── image2.txt
        ├── ...
        └── image367.txt

## Project Structure

The project is organized into the following directories and subdirectories:

- **Data**: Contains the dataset with subdirectories for images and labels.
- **Weights**: Stores the best-performing model weights obtained during training.
- **Results**: Holds the end results of the license plate detection system.
- **Model-Inference**: Provides example code and documentation for model inference.
- **Train-Val-Inference**: Includes code and instructions for inference during training and validation phases.
- **Evaluation Metrics**: Contains scripts and documentation for evaluating model performance, including metrics like precision, recall, and mAP.
 
## Training the Model

We used the YOLOv8 architecture for license plate detection. The model was fine-tuned based on a single class (license plates), and the 80 classes from the COCO dataset were removed in the YOLOv8 configuration file (yml).

### Training Parameters

- Image Size: 512x512 pixels
- Number of Epochs: 10 (adjust based on dataset size and convergence)
- Learning Rate: lr_scheduler
- Optimizer: smart_optimizer

### Training Process

The training process involved iterating through the dataset, feeding images and labels to the model, calculating loss, and optimizing the model using backpropagation. Training metrics were recorded for evaluation.

## Evaluation

Model performance was evaluated using various metrics, including object loss, class loss, precision, recall, and mAP (mean Average Precision). The results are shown below:

```
train/obj_loss     train/cls_loss    metrics/precision    metrics/recall    metrics/mAP_0.5    metrics/mAP_0.5:0.95    val/box_loss    val/obj_loss    val/cls_loss    x/lr0    x/lr1    x/lr2
0.014776           0                 0.80822              0.67873           0.7727             0.33475                 0.041708        0.010181       0               0.00208  0.00208  0.00208
```

## Conclusion

This project successfully fine-tuned the YOLOv8 model for license plate detection. The combination of a well-structured dataset, appropriate training parameters, and evaluation metrics has resulted in a robust license plate detection system.

For more detailed instructions and code examples, refer to the specific directories and README files within this project.



