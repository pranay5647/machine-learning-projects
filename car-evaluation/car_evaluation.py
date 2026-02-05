import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


def main():
    # Load data
    columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    df = pd.read_csv("car.data", header=None, names=columns)

    X = df.drop("class", axis=1)
    y = df["class"]

    # Train/test split (stratified because classes are imbalanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing: one-hot encode all categorical columns
    categorical_features = X.columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Model
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    # Pipeline = preprocessing + model (this is what we will save)
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", clf)
    ])

    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%\n")
    print("Classification report:\n")
    print(classification_report(y_test, y_pred))

    # Confusion matrix plot (saved to file)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Car Evaluation - Confusion Matrix (Random Forest)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    plt.close()

    # Save trained pipeline (preprocessor + model)
    with open("car_evaluation_model.pkl", "wb") as f:
        pkl.dump(model, f)

    print("\nSaved:")
    print("- car_evaluation_model.pkl")
    print("- confusion_matrix.png")


if __name__ == "__main__":
    main()
