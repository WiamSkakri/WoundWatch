# 🩹 WoundWatch

**WoundWatch** is an AI-powered web application that helps assess wound images for infection.

---

## 🔍 Features

- ✅ **Wound Classification**
  Classifies wound images as **healing** or **infected** using a fine-tuned ResNet-18 model.

- 🔥 **Grad-CAM Visualization**
  Visual explanation showing what parts of the image influenced the model's decision.

- 🎯 **Wound Segmentation**
  Uses a U-Net architecture to segment the wound region from the surrounding skin.

- 💻 **Interactive UI**
  Built with StreamLit for drag-and-drop image uploads and real-time AI inference.

---

## 🧠 Models

- **Classifier**: ResNet-18 pretrained on ImageNet, fine-tuned on labeled wound images.
- **Segmenter**: U-Net trained on the [Medetec wound dataset](https://www.kaggle.com/data)

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Woundwatch.git
cd Woundwatch
```

### Install dependencies

```bash
pip install -r requirements.txt
```

💬 Don't forget to activate your Python environment if you use one (e.g., `conda activate woundwatch` or `source venv/bin/activate`).

### Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## 📁 Project Structure

```bash
Woundwatch/
├── app/                    # StreamLit frontend
│   └── streamlit_app.py
├── classification/         # Classifier model + Grad-CAM
│   ├── train_classifier.py
│   └── model_resnet.py
├── segmentation/          # U-Net segmentation
│   ├── train_segmenter.py
│   ├── predict_mask.py
│   ├── dataset.py
│   └── unet.py
├── data/                  # Optional: sample images or masks
├── requirements.txt
└── README.md
```

## 📷 Demo

Upload a wound image and get:

* ✅ Diagnosis: Healing or Infected
* 🔥 Heatmap: Model attention via Grad-CAM
* 🎯 Segmentation: Mask of wound region

<!-- Optional image -->

---

## 💉 Notes

* This project is for research and demonstration purposes only.
* Model accuracy may vary depending on dataset labeling and domain variability.
* Clinical validation is required for medical deployment.

## 👨‍💻 Author

Wiam Skakri
@wiamskakri
wiamskakri.com

