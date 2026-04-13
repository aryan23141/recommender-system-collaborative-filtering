# 🎬 Hybrid Recommender System (Collaborative Filtering)

## 📌 Overview

This project implements a **hybrid movie recommendation system** using both **user-based** and **item-based collaborative filtering** techniques. The goal is to predict user ratings for unseen movies based on historical interactions.

The system is built **from scratch using NumPy**, without relying on high-level machine learning libraries.

---

## ⚙️ Features

* User-based Collaborative Filtering (using Pearson similarity)
* Item-based Collaborative Filtering (cosine similarity)
* Hybrid prediction model combining both approaches
* Top-K neighbor selection for better predictions
* Evaluation using **Normalized Mean Absolute Error (NMAE)**

---

## 🗂 Dataset

* MovieLens dataset (`u1.base` and `u1.test`)
* Contains user-item ratings

---

## 🧠 Methodology

### 1. User-Based Collaborative Filtering

* Computes similarity between users using **Pearson correlation**
* Predicts ratings based on similar users

### 2. Item-Based Collaborative Filtering

* Computes similarity between items using **cosine similarity**
* Predicts ratings based on similar items rated by the user

### 3. Hybrid Model

* Combines both approaches:

  [
  \hat{r} = \alpha \cdot r_{user} + (1 - \alpha) \cdot r_{item}
  ]

---

## 📊 Evaluation Metric

* **Mean Absolute Error (MAE)**
* **Normalized MAE (NMAE)**:

  [
  NMAE = \frac{MAE}{4}
  ]

---

## 🚀 Results

* The hybrid approach improves prediction accuracy by combining strengths of both methods.
* Final performance is reported using NMAE on the test dataset.

---

## 🛠 Tech Stack

* Python
* NumPy
* Pandas

---

## 📌 How to Run

```bash
python your_script.py
```

Make sure the dataset files (`u1.base`, `u1.test`) are in the same directory.

---

## 📈 Future Improvements

* Implement matrix factorization (SVD)
* Add implicit feedback handling
* Build a web-based interface for recommendations

---

## 👤 Author

Aryan Bundela
