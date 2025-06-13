# 📉 Linear Regression from Scratch

This project implements **Linear Regression** using **batch gradient descent**, coded entirely from scratch in Python. The dataset is synthetically generated using `sklearn.datasets.make_regression`, and all learning logic is manually implemented using `NumPy`.

---

## 🚀 Highlights

- Implements gradient descent for multivariate linear regression.
- Uses a manually defined cost function: **Mean Squared Error (MSE)**.
- Visualizes feature-target relationships and training cost convergence.
- Displays predictions alongside actual target values for evaluation.

---

## 📐 Cost Function

The cost is calculated using the **Mean Squared Error**:

$$
J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

Where:
- $m$ is the number of training examples  
- $\hat{y}^{(i)} = \mathbf{w} \cdot \mathbf{x}^{(i)} + b$ is the predicted value

---

## 🔁 Gradient Descent Updates

The parameters are updated as follows:

$$
\mathbf{w} := \mathbf{w} - \alpha \cdot \frac{\partial J}{\partial \mathbf{w}}
$$

$$
b := b - \alpha \cdot \frac{\partial J}{\partial b}
$$

Where:
- $\alpha$ is the learning rate

---

## 📦 Requirements

Install dependencies:

```bash
pip install numpy matplotlib pandas scikit-learn

---
```

## 👨‍💻 Author

Developed by a university student exploring machine learning and AI from scratch.

---
