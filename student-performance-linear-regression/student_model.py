import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    # 1. Load the dataset
    df = pd.read_csv("student-mat.csv", sep=";")

    # 2. Select relevant columns
    df = df[["G1", "G2", "G3", "studytime", "failures"]]

    # 3. Define features (X) and target (y)
    X = df[["G1", "G2", "studytime", "failures"]]
    y = df["G3"]

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=5
    )

    # 5. Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. Make predictions
    y_pred = model.predict(X_test)

    # 7. Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 8. Print results
    print("Model coefficients:")
    print(pd.Series(model.coef_, index=X.columns))

    print("\nIntercept:")
    print(model.intercept_)

    print("\nEvaluation metrics (test set):")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")


if __name__ == "__main__":
    main()


