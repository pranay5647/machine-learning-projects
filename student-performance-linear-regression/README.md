# Student Performance — Linear Regression

Predict students’ final grade (G3) using the UCI Student Performance dataset.

## Features used
- G1 (first period grade)
- G2 (second period grade)
- studytime
- failures

## Model
- Linear Regression (scikit-learn)
- Train/Test split: 75% / 25% (random_state=5)

## Results (test set)
- MAE ≈ 1.38
- RMSE ≈ 2.42
- R² ≈ 0.75

## How to run
1. Install dependencies:
   `pip install -r requirements.txt`
2. Run:
   `python student_model.py`
