#  SafeSteps – Hybrid CNN–Transformer Based Human Fall Detection

##  Overview

SafeSteps is a deep learning-based Human Fall Detection system designed for intelligent healthcare monitoring. The project uses a **Hybrid CNN–Transformer architecture** to detect fall and non-fall activities from wearable sensor data. By combining the feature extraction capability of Convolutional Neural Networks (CNN) with the sequence learning capability of the Transformer Encoder, the system automatically learns meaningful spatial and temporal patterns from time-series sensor signals. :contentReference[oaicite:0]{index=0}

The project is implemented using **Python**, **PyTorch**, and **Google Colab**, and is trained using the **SisFall Dataset**, which contains accelerometer and gyroscope recordings of daily activities and simulated falls. 

---

# Problem Statement

Falls are one of the leading causes of injury among elderly individuals and people with neurological or mobility disorders. Traditional fall detection approaches such as threshold-based methods and conventional machine learning algorithms often suffer from high false alarms, manual feature engineering, and limited ability to model complex movement patterns.

SafeSteps addresses these challenges by developing a Hybrid CNN–Transformer model that automatically extracts spatial and temporal features from wearable sensor data, improving classification performance while reducing the need for handcrafted features. 

---

#  Key Features

- Hybrid CNN–Transformer architecture for Human Fall Detection
- Automatic spatial feature extraction using CNN
- Temporal sequence learning using Transformer Encoder with self-attention
- Sliding Window sequence generation for time-series sensor data
- Data preprocessing and normalization using StandardScaler
- Binary classification of **Fall** and **Non-Fall** activities
- Performance evaluation using Accuracy, Precision, Training Loss and Confusion Matrix
- Implemented using PyTorch in Google Colab 

---

#  Dataset

This project uses the **SisFall Dataset**, a publicly available dataset developed for Human Fall Detection research.

The dataset contains:

- Accelerometer sensor data
- Gyroscope sensor data
- Daily Activities (D Files)
- Fall Activities (F Files)

The raw sensor data is preprocessed through data cleaning, normalization, and sliding-window sequence generation before being used to train the Hybrid CNN–Transformer model. 

> **Note:** The dataset is not included in this repository because it exceeds GitHub's file size limit.

Official dataset:
https://github.com/larocs/SisFall_dataset

---

#  Proposed Methodology

The proposed Human Fall Detection system follows these steps:

1. Load the SisFall Dataset.
2. Read accelerometer and gyroscope sensor data.
3. Perform data cleaning and preprocessing.
4. Normalize sensor values using StandardScaler.
5. Generate fixed-length sequences using the Sliding Window technique.
6. Extract local motion features using a CNN.
7. Learn long-term temporal dependencies using a Transformer Encoder.
8. Perform binary classification through a fully connected layer.
9. Predict the activity as **Fall** or **Non-Fall**. :contentReference[oaicite:5]{index=5}

---

#  Technologies Used

- Python
- PyTorch
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- Matplotlib :contentReference[oaicite:6]{index=6}

---

# How to Run

1. Clone this repository.

```bash
git clone https://github.com/Jinatsultana/SafeSteps.git
```

2. Open the notebook from the `notebooks` folder using Google Colab.

3. Download the SisFall Dataset from the official repository.

4. Upload the dataset when prompted in the notebook.

5. Run all cells sequentially to:
   - preprocess the dataset
   - train the Hybrid CNN–Transformer model
   - evaluate model performance
   - generate training plots

---

#  Results

The proposed Hybrid CNN–Transformer model successfully learned meaningful motion patterns from accelerometer and gyroscope sensor data.

During training:

- Training loss continuously decreased across epochs, indicating successful convergence.
- CNN effectively extracted local motion-related features.
- Transformer Encoder captured long-term temporal dependencies within sequential sensor data.
- The model was evaluated using Accuracy, Precision, Training Loss and Confusion Matrix.
- The hybrid architecture demonstrated improved learning capability by combining efficient feature extraction with powerful sequence learning. 

The generated evaluation plots are available in the **plots** folder.

---

#  Applications

- Elderly healthcare monitoring
- Assisted living systems
- Smart hospitals
- Wearable health devices
- Remote patient monitoring
- Emergency response systems :contentReference[oaicite:8]{index=8}

---

#  Future Scope

Future improvements include:

- Integration with wearable devices such as smartwatches and fitness bands
- IoT-based emergency notification systems
- Mobile and cloud deployment
- Edge AI implementation for low-latency inference
- Evaluation on larger real-world datasets
- Exploration of advanced architectures such as Vision Transformers and Graph Neural Networks :contentReference[oaicite:9]{index=9}

---

#  Conclusion

The Hybrid CNN–Transformer Based Human Fall Detection System demonstrates how deep learning can improve healthcare monitoring by combining CNN-based feature extraction with Transformer-based sequence learning. Trained on the SisFall Dataset after preprocessing, normalization, and sliding-window sequence generation, the model showed stable convergence and effective learning throughout training. The approach reduces manual feature engineering while improving representation of spatial and temporal information, making it a strong foundation for future AI-enabled healthcare and wearable monitoring systems. :contentReference[oaicite:10]{index=10}

---

# 👩‍💻 Author

**Jinat Sultana**

B.Tech Final Year Project – SafeSteps
