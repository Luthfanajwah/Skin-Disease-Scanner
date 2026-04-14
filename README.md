# Skin-Disease-Scanner
## Project Overview
Skin Disease Scanner is a deep learning based application designed to assist in the identification of various skin conditions. By leveraging advanced computer vision techniques, this project aims to provide an accessible and efficient tool for preliminary skin anomaly detection.

## Dataset
The model was trained and validated using a comprehensive collection of dermatological images.
1. Source 1 is from Mendeley Data Skin Disease Dataset. The dataset consists of 2,000+ high-quality images covering multiple categories of skin diseases, ensuring a robust foundation for feature extraction and classification.
2. Source 2 is from personal dataset thats consist of 40+ images covering multiples categories of skin deseases but different types than source 1.

## Technical Methodology
To achieve a balance between computational efficiency and high accuracy, both of datasets utilizes the following architecture MobileNetV2. MobileNetV2 was selected for its lightweight structure and use of inverted residual blocks, making it highly optimized for mobile and edge device deployment without compromising significant predictive performance. The approach involves Transfer Learning on the pre-trained MobileNetV2 weights, fine-tuned specifically for the nuances of dermatological patterns found in the dataset.
