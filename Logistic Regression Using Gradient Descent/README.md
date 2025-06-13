# 📌 Logistic Regression from Scratch

This project provides a Python implementation of the **logistic regression algorithm** from the ground up. It uses **NumPy** for numerical computations and **Matplotlib** for visualization. The goal is to demonstrate the fundamental mechanics of logistic regression - including the cost function and gradient descent - on a synthetic, linearly separable dataset.

---

## 🌟 Features

- 🚀 **Custom Implementation**: Core logistic regression logic without using `scikit-learn`’s `LogisticRegression`.
- 🧪 **Synthetic Data**: Uses `sklearn.datasets.make_classification` to create linearly separable 2D data.
- 📉 **Gradient Descent**: Implements batch gradient descent to optimize weights and bias.
- 📊 **Visualization**: Plots both:
  - The initial dataset
  - The decision boundary learned by the model

---

## 🛠️ Requirements

Make sure you have Python 3 installed along with the following libraries:

- `numpy`
- `matplotlib`
- `scikit-learn`
- `pandas`

Install dependencies using pip:

```bash
pip install numpy matplotlib scikit-learn pandas
```

---

## 🚀 Usage

1. Save the code as a Python file (e.g., `logistic_regression.py`)
2. Open your terminal or command prompt.
3. Navigate to the directory where you saved the file.
4. Run the script:

```bash
python logistic_regression.py
```

The script will display:
- A scatter plot of the initial dataset
- A plot showing the learned decision boundary with background probabilities

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

#### Gradient Descent

- `gradient_descent(...)` updates weights and bias using:

$$
w := w - \alpha \frac{\partial J(w, b)}{\partial w}
$$

$$
b := b - \alpha \frac{\partial J(w, b)}{\partial b}
$$

Where $\alpha$ is the learning rate.

### 3. Prediction and Decision Boundary

- `predict(X, w, b)`: Computes predicted probabilities.
- Uses `matplotlib.contourf` to draw the decision boundary where the model predicts $P(y=1|x)=0.5$.

---

## 📊 Output Visualizations

The script produces the following plots:

1. **Initial Data**: Scatter plot of the two classes.
2. **Decision Boundary**: The learned decision boundary with predicted probabilities visualized as a heatmap.

---

## 📚 Further Ideas

- Add L2 Regularization
- Try Stochastic or Mini-batch Gradient Descent
- Add learning curves
- Animate gradient descent steps
- Compare results with `scikit-learn`'s `LogisticRegression`

---

## 👨‍💻 Author

Developed by a university student exploring machine learning and AI from scratch.

---
