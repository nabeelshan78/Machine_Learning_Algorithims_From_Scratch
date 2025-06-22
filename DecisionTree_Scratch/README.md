# 🌳 MyDecisionTreeClassifier (from Scratch)

This is a lightweight and simple implementation of a **Decision Tree Classifier** using NumPy, built from scratch. It uses entropy and information gain to choose splits and builds the tree recursively.

---

## 🔍 Math Intuition

### 📘 Entropy

Entropy measures the uncertainty (impurity) in the class labels.

$$
H(y) = -\sum_{i=1}^{C} p_i \log_2(p_i)
$$

Where:
- \( C \) is the number of classes
- \( p_i \) is the proportion of samples in class \( i \)

---

### 🟰 Information Gain

Information Gain tells us how much entropy is reduced after splitting.

$$
IG = H(y) - \left( \frac{n_{\text{left}}}{n} \cdot H(y_{\text{left}}) + \frac{n_{\text{right}}}{n} \cdot H(y_{\text{right}}) \right)
$$

**Where:**
- `n` = total number of samples
- `n_left`, `n_right` = number of samples in the left and right child
- `H(y_left)`, `H(y_right)` = entropy of the left and right child nodes

---

## ✅ Results (on Iris Dataset)

![image](https://github.com/user-attachments/assets/62c201f2-fc15-4985-806b-b379771e67af)

---
