# Potato Disease Prediction using Convolutional Neural Networks (CNN)

## Objective
The goal of this project is to predict the disease status of potato leaves (Healthy, Early Blight, Late Blight) using images of the leaves. Convolutional Neural Networks (CNNs) are employed to classify the images into one of the three categories. The model is trained to recognize patterns in the images, and it achieves an impressive accuracy of **98%** in predicting the health condition of the potato plants.

## Methodology

### Data Collection
The dataset used for this project consists of images of potato leaves, categorized into three classes: **Healthy**, **Early Blight**, and **Late Blight**. This dataset can be found on various public platforms such as Kaggle or other agriculture-related datasets.

### Data Preprocessing
- **Image Resizing:** The images are resized to a fixed dimension to ensure uniformity.
- **Normalization:** Pixel values of the images are normalized to a range of 0 to 1 to help the model converge faster.
- **Augmentation:** Data augmentation techniques such as rotation, flipping, and zooming were applied to improve the generalization of the model and increase dataset diversity.

### CNN Model Architecture
- The model architecture is based on Convolutional Neural Networks (CNNs), which is particularly effective for image classification tasks.
- Layers used include convolutional layers, max-pooling layers, dropout layers for regularization, and fully connected layers for classification.
  
### Model Training
- The model is trained using a **categorical cross-entropy** loss function and the **Adam optimizer**.
- **Accuracy** and **loss** metrics are monitored during training to evaluate the model's performance.
- The model achieved a remarkable **98% accuracy** on the test set, demonstrating excellent performance.

### Training and Validation Plots
- **Accuracy Plot:** The training and validation accuracy over each epoch are plotted to visualize how well the model performs and generalizes.
- **Loss Plot:** The training and validation loss are also plotted to observe the convergence of the model during training.

## Key Results
- Achieved **98% accuracy** in classifying potato leaf images into three categories: Healthy, Early Blight, and Late Blight.
- Plotted training and validation accuracy, as well as loss curves, to visualize the model's performance and convergence.
- CNN model demonstrated high accuracy and robust generalization to unseen data, indicating its effectiveness in agricultural disease detection.

## Tools & Libraries
- **Python**
- **Keras / TensorFlow** (for building and training the CNN model)
- **Matplotlib** (for plotting training and validation accuracy and loss)
- **NumPy** (for numerical operations)

## How to Run
1. Clone the repository:  
   `git clone https://github.com/Harsha-S-Naik/Potato_Disease_Prediction.git`
   
2. Install the required libraries:  
   `pip install -r requirements.txt`
   
3. Run the Jupyter Notebook train the model and visualize the results:  
   `jupyter notebook`

4.To run
`uvicorn app:app --reload`

## Preview
![potato](https://github.com/user-attachments/assets/9fa501bb-a62c-4ff3-b27f-9f3b7c302bad)



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


