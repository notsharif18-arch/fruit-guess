# 🍎 FruitVision AI

A deep learning computer vision project that classifies fruits using **TensorFlow**, **MobileNetV2**, and **Streamlit**.

---

## Features

- 🍌 15+ Fruit Classes
- 🧠 MobileNetV2 Transfer Learning
- 🌐 Streamlit Web App
- 📷 Webcam Detection
- 🖼️ Image Upload Prediction
- 📊 Training & Validation
- 📈 Model Evaluation
- ⚡ One-click Setup
- 🎯 Top-5 Predictions

---

## Tech Stack

- Python 3.11
- TensorFlow
- Keras
- OpenCV
- Streamlit
- NumPy
- Pillow
- Matplotlib

---

## Project Structure

```text
fruit-guess/
│
├── app.py
├── train.py
├── predict.py
├── webcam.py
├── evaluate.py
├── download_dataset.py
├── setup.bat
├── run.bat
├── requirements.txt
│
├── src/
│   ├── config.py
│   ├── model.py
│   └── dataloader.py
│
├── models/
├── dataset/
└── assets/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/notsharif18-arch/fruit-guess.git
cd fruit-guess
```

Run:

```bash
setup.bat
```

---

## Train the Model

```bash
python train.py
```

---

## Launch Web App

```bash
run.bat
```

or

```bash
streamlit run app.py
```

---

## Webcam Detection

```bash
python webcam.py
```

---

## Evaluate

```bash
python evaluate.py
```

---

## Screenshots

Add screenshots here after training.

---

## Future Improvements

- Better UI
- More fruit classes
- TensorBoard support
- Fine tuning
- ONNX export

---

## License

MIT License
