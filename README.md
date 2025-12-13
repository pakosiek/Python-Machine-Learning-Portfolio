# Python Deep Learning Portfolio

A comprehensive collection of Deep Learning projects demonstrating a progression from basic Neural Networks (MLP) to advanced Computer Vision (CNN) and large-scale classification tasks.

## Project Structure

### 1. [01_Neural_Network_Foundations.ipynb](01_Neural_Network_Foundations.ipynb) 
**Focus:** Core concepts of Neural Networks.
* **Tasks:** Solving non-linear problems (XOR) and synthetic data classification (Blobs).
* **Key Concepts:** Decision boundaries, Activation functions (Sigmoid, ReLU), Dense layers.

### 2. [02_Medical_Diagnostics_and_Tabular_Data.ipynb](02_Medical_Diagnostics_and_Tabular_Data.ipynb) 
**Focus:** Data Science & Deep Learning for Tabular Data.
* **Diabetes Prediction:** Binary classification on Pima Indians Diabetes Dataset using BatchNormalization and Dropout.
* **Iris Classification:** Comparison of Binary vs. Multi-class classification strategies.
* **Student Performance:** Fetching data via **UCI Machine Learning Repository API** (`ucimlrepo`) to predict student grades.

### 3. [03_Computer_Vision_Handwritten_Digits.ipynb](03_Computer_Vision_Handwritten_Digits.ipynb) 
**Focus:** Interactive Computer Vision (MNIST).
* **Model:** Multi-Layer Perceptron (MLP) for digit recognition.
* **Feature:** **Interactive HTML5 Canvas** allowing users to draw digits directly in the Jupyter Notebook and get real-time predictions.
* **Analysis:** Error analysis visualizing misclassified digits.

### 4. [04_Advanced_Vision_Object_Recognition.ipynb](04_Advanced_Vision_Object_Recognition.ipynb) 
**Focus:** Convolutional Neural Networks (CNN) & Scaling.
* **Part 1 (CIFAR-10):** Classifying 10 distinct object categories (Airplanes, Cats, Ships, etc.) using a custom CNN architecture.
* **Part 2 (CIFAR-100):** Scaling the architecture to handle **100 fine-grained classes**.
* **Tech Stack:** Deep VGG-style CNN, Heavy Regularization (Dropout 0.5), Batch Normalization, and Early Stopping.

---

## 🛠️ Technologies Used

* **Core:** `Python`, `NumPy`, `Pandas`
* **Deep Learning:** `TensorFlow`, `Keras`
* **Visualization:** `Matplotlib`, `Seaborn`
* **Data Processing:** `Scikit-Learn`, `OpenCV`, `Pillow`
* **Data Source:** `UCIMLRepo API`

## 🚀 How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/pakosiek/Python-Deep-Learning-Portfolio.git](https://github.com/pakosiek/Python-Deep-Learning-Portfolio.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Open the notebooks in Jupyter Lab, Jupyter Notebook, or Google Colab.

---
*Created by Damian Pakosiński*
