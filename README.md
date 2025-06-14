# 🧠 Handwritten Equation Solver & Graphing Web App

This project is a web-based application that detects handwritten **algebraic equations**, solves them, and generates corresponding **graphs**, providing an intuitive interface for math learners and educators.

## 🚀 Features

- ✍️ Recognizes handwritten algebraic equations using a **CNN model**
- 🧮 Parses and solves the equation in real-time
- 📈 Displays dynamic graph visualizations of the equations
- 🖥️ Built with **Python**, leveraging computer vision and deep learning

## 🧰 Tech Stack

- **Frontend:** HTML, CSS, JavaScript (optional Dash or Flask integration)
- **Backend:** Python
- **ML Framework:** TensorFlow / PyTorch (CNN for equation recognition)
- **Graphing:** Matplotlib / Plotly
- **Equation Parsing:** SymPy

## 🧠 Model Details

- Convolutional Neural Network (CNN) trained to identify digits and math symbols
- Dataset: Custom-prepared dataset of handwritten equations (can be extended using tools like Roboflow)
- Preprocessing includes binarization, normalization, and contour detection

## 📊 Workflow

1. **Input:** User draws/writes an equation on a digital canvas  
2. **Recognition:** Image is processed and passed to CNN  
3. **Solution:** Parsed using SymPy to solve the equation  
4. **Graphing:** Solution and its graph are displayed instantly 
