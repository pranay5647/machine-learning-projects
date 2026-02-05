import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    df = pd.read_csv("student-mat.csv", sep=";")
    df = df[["G1", "G2", "G3", "studytime", "failures"]]

    x = df[["G1", "G2", "studytime", "failures"]]
    y = df["G3"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=5
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Model coefficients:")
    print(pd.Series(model.coef_, index=x.columns))

    print("\nIntercept:")
    print(model.intercept_)

    print("\nEvaluation metrics (test set):")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

    # Plot: Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([0, 20], [0, 20])
    plt.xlabel("Actual G3")
    plt.ylabel("Predicted G3")
    plt.title("Actual vs Predicted Final Grades")
    plt.tight_layout()
    plt.show()

    with open("student_model.pkl", "wb") as f:
        pkl.dump(model, f)


if __name__ == "__main__":
    main()
