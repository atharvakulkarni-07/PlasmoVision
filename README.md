# PlamoVision: Malaria Diagnosis using Deep Learning

**PlamoVision** is a deep learning project that uses Computer Vision to automatically diagnose whether a given cell image is parasitized (infected with malaria) or uninfected. This project leverages Convolutional Neural Networks (CNNs) to build an image classifier capable of distinguishing between parasitized and uninfected cells, aiming to assist in malaria diagnosis and potentially streamline the medical screening process.

**Please find the Web Application Link Here**: [APP LINK](https://plasmovision-malaria.streamlit.app/)

## 📜 Project Overview

Malaria is a life-threatening disease that is transmitted to humans through the bites of infected mosquitoes. Early detection and accurate diagnosis are crucial in reducing its mortality rate. Typically, malaria diagnosis is performed manually by skilled technicians analyzing blood smears under a microscope, a time-consuming and error-prone task. **PlamoVision** aims to automate this process using a deep learning model.

## 🔍 Problem Statement

The goal of this project is to develop a deep learning-based image classifier that can accurately categorize cell images as either:
1. **Parasitized** - Cells infected with malaria parasites.
2. **Uninfected** - Healthy cells free of infection.

This solution can assist in reducing diagnostic time and help in areas with limited access to skilled healthcare professionals.

## 📂 Dataset

The dataset used for training and testing is the **Malaria Cell Images Dataset**, which contains a collection of cell images labeled as either parasitized or uninfected. The dataset has been divided into training and validation sets for model evaluation.

- **Total Images**: ~27,000
- **Parasitized Cells**: ~13,800
- **Uninfected Cells**: ~13,800

You can download the dataset from [here](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria).

## 🛠️ Methodology

The methodology followed in this project involves the following steps:

1. **Data Preprocessing**
   - Resizing images to a uniform shape.
   - Data Augmentation (Rotation, Zoom, Horizontal Flip) to enhance the training set.
  
2. **Model Architecture**
   - A Convolutional Neural Network (CNN) model built with Keras.
   - The architecture consists of several convolutional and pooling layers, followed by dense layers for classification.
  
3. **Training and Validation**
   - Training the model using the **Binary Cross-Entropy** loss function.
   - Optimized using **Adam** optimizer.
  
4. **Model Evaluation**
   - Accuracy, Precision, and Recall are calculated on the validation set.
   - Confusion Matrix to visualize the classification results.

## 💡 Key Features

- **Automatic Diagnosis**: Classifies cell images into two categories: parasitized or uninfected.
- **CNN-based Image Classification**: Uses convolutional layers to extract features and predict labels.
- **Data Augmentation**: Uses data augmentation techniques to avoid overfitting and improve model generalization.
- **User-Friendly Code**: Well-structured and easy-to-understand code, making it suitable for learning and experimentation.

## 📊 Results

- **Training Accuracy**: Achieved high accuracy on the training dataset.
- **Validation Accuracy**: Good performance on the validation set, indicating effective generalization.

### Example Confusion Matrix:

|                | Predicted Parasitized | Predicted Uninfected |
|----------------|-----------------------|----------------------|
| **True Parasitized** | 93%                   | 7%                   |
| **True Uninfected**  | 8%                    | 92%                  |

## 🚀 How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/PlamoVision.git
   cd PlamoVision
   ```

2. **Install Required Libraries**:
   Make sure you have Python 3.8 or later. Then, install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   Open the `Malaria_Diagnosis_file.ipynb` in Jupyter Notebook or Jupyter Lab to train and evaluate the model.


## 🧰 Tech Stack

- **Programming Language**: Python
- **Libraries**: TensorFlow, Keras, NumPy, Matplotlib
- **Development Environment**: Jupyter Notebook

## 🤝 Contributing

Feel free to fork this repository and make improvements. Pull requests are welcome!

## 💬 Contact

For any questions or collaboration, feel free to reach out:

- **Email**: [atharvakulkarni.official@gmail.com](url)
- **LinkedIn**: [https://www.linkedin.com/in/atharva-kulkarni-3b13a3255/](url)
