# 📌 Logistic Regression from Scratch

This project provides a Python implementation of the **logistic regression algorithm** from the ground up. It uses **NumPy** for numerical computations and **Matplotlib** for visualization. The goal is to demonstrate the fundamental mechanics of logistic regression - including the cost function and gradient descent - on a synthetic, linearly separable dataset.

---

## 🌟 Features

- 🚀 **Custom Implementation**: Core logistic regression logic without using `scikit-learn`’s `LogisticRegression`.
- 🧪 **Synthetic Data**: Uses `sklearn.datasets.make_classification` to create linearly separable 2D data.
- 📉 **Gradient Descent**: Implements batch gradient descent to optimize weights and bias.
- 📊 **Visualization**: Plots both:
  - The initial dataset
  - Cost function over iterations
  - The decision boundary learned by the model

---

## 🛠️ Requirements

Make sure you have Python 3 installed along with the following libraries:

- `numpy`
- `matplotlib`
- `scikit-learn`
- `pandas`
- `math`

---

## ⚙️ Code Structure

### 1. Data Generation and Visualization

- `make_classification`: Generates a 2D synthetic binary classification dataset.
- `plt.scatter`: Visualizes the dataset using a scatter plot with class labels.

### 2. Core Logistic Regression Functions

#### Cost Function

- `compute_cost(X, y, w, b)` calculates the binary cross-entropy loss:

$$
J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f_{w,b}(x^{(i)})) + (1 - y^{(i)}) \log(1 - f_{w,b}(x^{(i)})) \right]
$$

where

$$
f_{w,b}(x) = \sigma(w \cdot x + b)
$$

and $\sigma$ is the sigmoid function.

#### Gradient Computation

- `compute_gradient(X, y, w, b)` computes gradients of the cost function with respect to weights and bias.
![image](https://github.com/user-attachments/assets/5f341991-17dc-4e05-8091-d3eb1dd7d87c)
![image](https://github.com/user-attachments/assets/6f26718b-17f2-4517-8128-cd980051c46f)
![image](https://github.com/user-attachments/assets/c98bf50d-ad6c-40e3-8dab-7a2ee5d05818)




#### Gradient Descent

- `gradient_descent(...)` updates weights and bias using:

$$
w := w - \alpha \frac{\partial J(w, b)}{\partial w}
$$

$$
b := b - \alpha \frac{\partial J(w, b)}{\partial b}
$$

Where $\alpha$ is the learning rate.

### 3. Cost Function 
![image](https://github.com/user-attachments/assets/d6f610ca-eebe-4c78-ba5f-d5ebff9977e5)


### 4. Prediction and Decision Boundary

- `predict(X, w, b)`: Computes predicted probabilities.
- Uses `matplotlib.contourf` to draw the decision boundary where the model predicts $P(y=1|x)=0.5$.
![image](https://github.com/user-attachments/assets/2ef3219b-90c1-4bd1-beae-1d8a2ec820e2)

---

## 👨‍💻 Author

- [Nabeel Shan](https://github.com/nabeelshan78)

A university student exploring machine learning and AI from scratch.

---
