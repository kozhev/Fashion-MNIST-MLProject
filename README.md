# Fashion-MNIST — Multi-Class Classification with a Hand-Written KNN

## תיאור הבעיה

מטרת הפרויקט היא לבצע סיווג אוטומטי של תמונות פריטי לבוש. כל תמונה מיוצגת כתמונה בגודל 28×28 פיקסלים, והמודל נדרש לזהות לאיזו מתוך 10 קטגוריות אופנה היא שייכת.

Multi-Class Classification - חלוקה של המידע ל10 קטגוריות שונות.

- **Input:** תמונה
- **Output:** אחת מתוך 10 מחלקות

<https://www.kaggle.com/datasets/zalando-research/fashionmnist>

---

## Overview

Computer-vision track project. A K-Nearest-Neighbours
classifier implemented from scratch in NumPy, applied to the
[Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
dataset: 28×28 greyscale clothing images sorted into 10 categories.

Quality metric: **macro-average F1**.


## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Running

```bash
.venv\Scripts\jupyter notebook fashion_mnist_knn.ipynb
```

The notebook runs the whole flow: loading, feature engineering, the grid search,
final training and test-set evaluation. Figures are written to `outputs/figures/`.

## Files

The notebook is self-contained — the configuration, data loading, feature
engineering, the KNN classifier, the grid search and every figure are all defined
in its own cells.

| File | Contents |
|---|---|
| `fashion_mnist_knn.ipynb` | The submission notebook — the entire project |
| `archive.zip` | The Kaggle download (not in git; see Setup) |
| `data/` | The two CSVs, extracted from the archive |
| `outputs/` | Figures and the grid-search results table |

### Notebook layout

| Part | Contents | Assignment |
|---|---|---|
| 1 | The problem, the data, the quality metric | Part 1 |
| 2 | Feature engineering — scaling and PCA | Part 2 |
| 3 | The KNN classifier, written from scratch | Part 3 |
| 4 | Grid search with 5-fold cross-validation | Part 6a (bonus) |
| 5 | Training the winning configuration on all the data | Part 4 |
| 6 | Prediction and evaluation on the test set | Part 5 |
| 7 | Explainability — why each prediction was made | Part 6c (bonus) |

## Tuning the run

Settings live next to the step they control, in a short settings cell at the top of
each part rather than in one block at the top of the notebook:

- **Part 1** — `MAX_TRAIN_SAMPLES`: `None` uses all 60,000 training rows; set it
  to `5000` for a fast development run. The subset is stratified, so the class
  balance is preserved. This is a size limit, not a second train/test split.
  `MAX_TEST_SAMPLES` does the same for the test set, and `RANDOM_SEED` fixes every
  random choice in the notebook.
- **Part 2** — `DEFAULT_SCALING` and `DEFAULT_PCA_COMPONENTS` for the worked
  examples, plus `PCA_CANDIDATES`, the component counts the search will try.
- **Part 3** — `DEFAULT_K`, `DEFAULT_METRIC`, `DEFAULT_WEIGHTS` for a
  `KNNClassifier` built without arguments, and `PREDICT_BATCH_SIZE`, which bounds
  peak memory during prediction.
- **Part 4** — `CV_FOLDS`, `GRID_SEARCH_SAMPLES` (training rows used inside the
  search, which runs the full cartesian product once per fold), and `PARAM_GRID`,
  the search space over scaling, PCA components, `k`, distance metric and vote
  weighting.
