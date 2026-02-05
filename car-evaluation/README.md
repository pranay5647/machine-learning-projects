# Car Evaluation (Classification) — Random Forest

Multiclass classification project using the UCI Car Evaluation dataset.

## Dataset
Features:
- buying, maint, doors, persons, lug_boot, safety

Target:
- class ∈ {unacc, acc, good, vgood}

All features are categorical, so one-hot encoding is required.

## Approach
- Train/test split: 80/20 with stratification
- Preprocessing: OneHotEncoder (handle_unknown="ignore")
- Model: RandomForestClassifier (n_estimators=300)

## Results (test set)
- Accuracy ≈ 97–98%
- Balanced performance across classes (including rare `good` and `vgood`)

## Diagnostics
A confusion matrix is generated and saved as:
- `confusion_matrix.png`

## How to run
```bash
pip install -r requirements.txt
python car_evaluation.py
