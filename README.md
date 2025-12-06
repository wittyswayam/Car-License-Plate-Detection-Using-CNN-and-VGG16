## 🚗 Vehicle License Plate Detector: Transfer Learning with VGG16 (Bounding Box Regression)

This repository details a deep learning project dedicated to **Vehicle License Plate Detection**. Unlike image classification, this task requires **object localization**, meaning the model must not only recognize an object but also draw a precise box around it.

The solution leverages a Convolutional Neural Network (CNN) foundation, specifically using the **VGG16** architecture via **Transfer Learning** and a custom head for **Bounding Box Regression**.

***

### ✨ Technical Features

* **Transfer Learning with VGG16:** We utilize the **VGG16** model, pre-trained on the massive **ImageNet** dataset, as a fixed **feature extractor**. This technique is crucial: it allows the model to benefit from millions of images it has already seen, extracting highly generalizable visual features (edges, textures, shapes), which are then applied to the specialized task of license plate detection, resulting in faster convergence and better performance with limited training data.
* **Bounding Box Regression:** The objective is to predict four continuous numerical values representing the normalized coordinates of the license plate within the image. The model's final output layer consists of **4 units** with **sigmoid activation**, predicting the scaled coordinates: `(xmin, ymin, xmax, ymax)`. This is a regression problem, not a classification problem.
* **Annotation Handling:** The notebook includes custom logic to process standard **PASCAL VOC**-formatted XML annotation files. This parsing step is vital for extracting the ground-truth pixel coordinates and scaling them correctly to match the size of the input images.

***

### 🛠️ Project Setup & Dependencies

To execute the `vehicle-license-plate-detection-cnn-vgg16.ipynb` notebook, ensure you have a Python environment with the following libraries installed:

* **Deep Learning:** `tensorflow` / `keras`
* **Data Processing:** `numpy`, `pandas`
* **Image Processing:** `opencv-python` (`cv2`)
* **XML Parsing:** `lxml`
* **Visualization:** `matplotlib`, `seaborn`

***

### 📊 Dataset and Preprocessing

The project utilizes a car plate detection dataset that follows a standard computer vision format:

1.  **Image Files** (JPEG).
2.  **Annotation Files** (XML, PASCAL VOC format).

**Preprocessing Steps Explained:**

* **Standardized Input Size:** All images are uniformly resized to **224x224 pixels**. This fixed size is required by the VGG16 architecture and ensures consistent input for the CNN.
* **Pixel Normalization:** Image pixel values (0-255) are normalized to the range of **0 to 1** by dividing by 255.0. This improves model training stability.
* **Coordinate Scaling:** The crucial preprocessing step for bounding boxes is to scale the ground-truth coordinates. The original PASCAL VOC coordinates are absolute pixel values based on the original image size. The code recalculates these coordinates to fit the new **224x224** dimension and normalizes them to the **0 to 1** range. This is essential because the model is trained to predict these normalized, scaled values, which are later denormalized during prediction to draw the box on the original image.

***

### 🧠 Model Architecture 

[Image of VGG16 Architecture]


The model is constructed in two primary parts:

1.  **VGG16 Base (Feature Extractor):**
    * The VGG16 model is loaded with pre-trained **ImageNet weights** but without its original top classification layer (`include_top=False`).
    * The base model is explicitly **frozen** (`vgg.trainable = False`). Freezing prevents the powerful, pre-learned filters from being randomly altered by the gradients during the training of the new regression head, preserving their high-level feature extraction capabilities.

2.  **Custom Regression Head:**
    * The output feature maps from the VGG16 base are **flattened** into a single vector.
    * This is followed by a simple Deep Neural Network (DNN) sequence:
        * A **Dense** layer (128 units, `relu` activation).
        * A **Dropout** layer (0.3) for regularization, which helps prevent overfitting.
    * The **Output Layer** is a **Dense layer with 4 units** and a **`sigmoid` activation function**. The sigmoid function guarantees the output values are between 0 and 1, perfectly matching the normalized target bounding box coordinates the model was trained on.

**Compilation Details:**

* **Optimizer:** `Adam`
* **Loss Function:** **Mean Squared Error (`'mse'`)** is used as the loss function. MSE calculates the average squared difference between the predicted coordinate values and the true coordinate values, making it the standard choice for continuous-value regression tasks.

***

### 📈 Training and Performance

The model was trained for **50 epochs**.

| Metric | Value (End of 50 Epochs) | Context |
| :--- | :--- | :--- |
| **Training Loss (MSE)** | ~0.0008 | Very low MSE indicates the model fits the training data well. |
| **Validation Loss (MSE)** | ~0.0182 | Measures performance on unseen data, providing a realistic estimate of the model's generalization ability. |
| **Final Test Score** | **63.64%** | Accuracy is calculated based on how closely the predicted box overlaps with the ground truth box (though a dedicated IoU metric would be more appropriate—see Future Works). |

The difference between the Training Loss and Validation Loss suggests a degree of **overfitting**. While the model has learned the bounding box locations, future work should focus on regularization or data augmentation to close this generalization gap.

***

### 💡 Future Works

The following enhancements and extensions are recommended for improving model robustness and performance:

 **Implement IoU (Intersection over Union) as a Metric:**
    * Currently, performance is evaluated using raw MSE and a simple "accuracy" metric. **IoU** is the gold standard for object detection. Implementing IoU as a custom Keras metric would provide a much more intuitive and accurate measure of bounding box localization performance.

  **Explore State-of-the-Art Architectures:**
    * While VGG16 is a solid baseline, modern object detection models like **YOLO (You Only Look Once)**, **SSD (Single Shot MultiBox Detector)**, or    **EfficientDet** are designed for faster inference and better accuracy. Implementing a more advanced architecture would likely yield superior results.

 **Integrate an OCR Module:**
    * The current project only detects the plate. The next logical step is to combine this detection with an **Optical Character Recognition (OCR)** module (e.g., using a separate CNN, Tesseract, or a CTC-based model) to read and transcribe the characters on the detected license plate.

