## AI Model Training

This repository contains scripts and configuration files used to train the AI models responsible for gesture recognition in the "Rock, Paper, Scissors" game.

### Implemented Models

- **Classic CNN Models**  
  Simple convolutional neural networks trained from scratch on a custom dataset of hand gesture images.

- **MediaPipe-Based Model**  
  Uses Google’s MediaPipe framework for real-time hand detection and landmark extraction, followed by gesture classification.

- **Transfer Learning with ResNet**  
  A model based on a pre-trained ResNet architecture, fine-tuned for the gesture classification task. This approach achieved the highest accuracy in the tests.

- **Segmentation with U-Net**  
  Utilizes image segmentation to isolate the hand region using a U-Net architecture, followed by classification based on segmented images. This approach showed strong generalization on diverse input data.

### Summary

Each model was trained and evaluated for performance and generalization ability.  
- **ResNet** performed best in terms of classification accuracy.  
- **MediaPipe** and **U-Net**-based models offered great robustness to noisy or atypical data.

