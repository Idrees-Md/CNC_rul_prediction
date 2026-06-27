from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "cnc_rul_dataset.csv")

df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()

print("Dataset loaded successfully")
print(df.head())
print(df.columns)

FEATURE_COLUMNS = ["Temperature_C", "Vibration_g", "Current_A", "RPM"]
TARGET_COLUMN = "RUL_Hours"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

MAE = mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
R2 = r2_score(y_test, y_pred)

non_zero_y_test = y_test.replace(0, np.nan)
MAPE = np.nanmean(np.abs((non_zero_y_test - y_pred) / non_zero_y_test)) * 100

print("Model trained successfully")
print("MAE:", MAE)
print("RMSE:", RMSE)
print("R2:", R2)
print("MAPE:", MAPE)

live_index = 0


def get_health_status(rul):
    if rul >= 7000:
        return {
            "status": "Good",
            "recommendation": "Machine is healthy. Continue regular monitoring."
        }
    elif rul >= 4000:
        return {
            "status": "Moderate",
            "recommendation": "Schedule inspection during the next maintenance window."
        }
    elif rul >= 1800:
        return {
            "status": "Warning",
            "recommendation": "Maintenance needed soon. Inspect spindle, bearings, and lubrication."
        }
    else:
        return {
            "status": "Critical",
            "recommendation": "Immediate maintenance required. Reduce load and inspect machine."
        }


@app.route("/api/live-data", methods=["GET"])
def live_data():
    global live_index

    try:
        row = df.iloc[live_index]

        time_hours = round(float(row["Time_Hours"]), 2)
        temperature = round(float(row["Temperature_C"]), 2)
        vibration = round(float(row["Vibration_g"]), 4)
        current = round(float(row["Current_A"]), 2)
        rpm = round(float(row["RPM"]), 2)
        actual_rul = round(float(row["RUL_Hours"]), 2)

        input_data = pd.DataFrame([{
            "Temperature_C": temperature,
            "Vibration_g": vibration,
            "Current_A": current,
            "RPM": rpm
        }])

        predicted_rul = round(float(model.predict(input_data)[0]), 2)

        health = get_health_status(predicted_rul)

        live_index = (live_index + 1) % len(df)

        return jsonify({
            "time_hours": time_hours,
            "temperature": temperature,
            "vibration": vibration,
            "current": current,
            "rpm": rpm,
            "actual_rul": actual_rul,
            "rul": predicted_rul,
            "status": health["status"],
            "recommendation": health["recommendation"],
            "index": live_index
        })

    except Exception as e:
        print("Live data error:", str(e))
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        temperature = float(data.get("temperature"))
        vibration = float(data.get("vibration"))
        current = float(data.get("current"))
        rpm = float(data.get("rpm"))

        input_data = pd.DataFrame([{
            "Temperature_C": temperature,
            "Vibration_g": vibration,
            "Current_A": current,
            "RPM": rpm
        }])

        predicted_rul = round(float(model.predict(input_data)[0]), 2)
        health = get_health_status(predicted_rul)

        return jsonify({
            "rul": predicted_rul,
            "status": health["status"],
            "recommendation": health["recommendation"]
        })

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model": "Random Forest Regression",
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "mae": round(float(MAE), 2),
        "rmse": round(float(RMSE), 2),
        "r2": round(float(R2), 4),
        "mape": str(round(float(MAPE), 2)) + "%"
    })


@app.route("/api/dataset-info", methods=["GET"])
def dataset_info():
    return jsonify({
        "rows": len(df),
        "columns": list(df.columns),
        "message": "Dataset loaded successfully"
    })


@app.route("/api/analytics", methods=["GET"])
def analytics():
    feature_importance = {}

    for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_):
        feature_importance[feature] = round(float(importance), 4)

    actual_values = [round(float(value), 2) for value in y_test.head(30)]
    predicted_values = [round(float(value), 2) for value in y_pred[:30]]

    return jsonify({
        "feature_importance": feature_importance,
        "actual": actual_values,
        "predicted": predicted_values,
        "mae": round(float(MAE), 2),
        "rmse": round(float(RMSE), 2),
        "r2": round(float(R2), 4),
        "mape": str(round(float(MAPE), 2)) + "%"
    })


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "CNC RUL Prediction Backend is running",
        "live_endpoint": "/api/live-data",
        "predict_endpoint": "/api/predict",
        "dataset_info": "/api/dataset-info",
        "model_info": "/api/model-info",
        "analytics": "/api/analytics"
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)