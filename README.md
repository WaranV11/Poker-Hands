# Poker Hand Strength Predictor

A machine learning project that classifies poker hands using multiple models, with a custom-built Artificial Neural Network (ANN) achieving **99.31% accuracy** on a dataset of over 1 million hands.

---

## Results

| Model | Test Accuracy |
|---|---|
| Artificial Neural Network (ANN) | **99.31%** |
| Decision Tree | ~63% |
| Gaussian Naive Bayes | ~50% |

The ANN dramatically outperformed classical approaches, demonstrating the effectiveness of neural networks for structured card-data classification.

---

## What it does

Given 5 playing cards (suit + rank of each), the model predicts the poker hand class:

| Class | Hand |
|---|---|
| 0 | Nothing in hand |
| 1 | One pair |
| 2 | Two pairs |
| 3 | Three of a kind |
| 4 | Straight |
| 5 | Flush |
| 6 | Full house |
| 7 | Four of a kind |
| 8 | Straight flush |
| 9 | Royal flush |

---

## Dataset

- **Source:** [UCI Machine Learning Repository — Poker Hand Dataset](https://archive.ics.uci.edu/dataset/158/poker+hand)
- **Size:** ~1,025,010 hands
- **Split:** 70% training (717,507 samples) / 30% testing (307,503 samples)
- **Features:** 10 input features — suit and rank for each of the 5 cards

---

## Approach

1. **Data loading & exploration** — fetched directly from UCI via `ucimlrepo`, visualized class distribution (heavily imbalanced — most hands are "nothing" or "one pair")
2. **Preprocessing** — applied `StandardScaler` normalization before feeding into models that require scaled input
3. **Baseline models** — trained Decision Tree and Gaussian Naive Bayes to establish benchmarks
4. **ANN** — built a multi-layer perceptron (`MLPClassifier`) with ReLU activation and 100 hidden units; evaluated with confusion matrix and accuracy score
5. **Evaluation** — compared all models using test accuracy and confusion matrix heatmaps

---

## Tech Stack

- Python
- scikit-learn (MLPClassifier, DecisionTreeClassifier, GaussianNB, StandardScaler)
- Pandas, NumPy
- Matplotlib, Seaborn
- ucimlrepo

---

## How to run

1. Open `Poker_Hands.ipynb` in Google Colab or Jupyter
2. Run all cells in order
3. The notebook will automatically fetch the dataset and train all models

No manual data download needed — the dataset is fetched programmatically via `ucimlrepo`.
