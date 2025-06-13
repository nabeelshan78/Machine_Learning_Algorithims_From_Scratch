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

## Gradients Calculation
Cost Function

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

Where:
$\hat{y}^{(i)} = w^T x^{(i)} + b = \sum_{j=1}^{n} w_j x_j^{(i)} + b$
- \( m \) is the number of training examples
- \( n \) is the number of features

Derivative w.r.t w  

$$
\frac{\partial J(w, b)}{\partial w_j} 
= \frac{1}{2m} \sum_{i=1}^{m} 2 \left( \hat{y}^{(i)} - y^{(i)} \right) \cdot \frac{\partial \hat{y}^{(i)}}{\partial w_j}
$$

So the gradient becomes:

$$
\frac{\partial J(w, b)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) x_j^{(i)}
$$

$$
\frac{\partial J(w, b)}{\partial w} 
= \frac{1}{2m} \sum_{i=1}^{m} 2 \left( (w^T x^{(i)} + b) - y^{(i)} \right) x^{(i)}
= \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) x^{(i)}
$$

Derivative w.r.t b  

$$
\frac{\partial J(w, b)}{\partial b} 
= \frac{1}{2m} \sum_{i=1}^{m} 2 \left( \hat{y}^{(i)} - y^{(i)} \right) \cdot \frac{\partial \hat{y}^{(i)}}{\partial b}
$$

$$
\frac{\partial J(w, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)
$$


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

## Cost over Iterations
![image](https://github.com/user-attachments/assets/42e51397-acf6-475d-b838-b3dc7bde6df3)

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
