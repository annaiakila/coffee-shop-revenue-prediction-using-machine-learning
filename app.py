# app.py
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd

# Try multiple loaders for the model for robustness
def load_model(path):
    # Try joblib, pickle (latin1), then dill
    errs = []
    try:
        import joblib
        m = joblib.load(path)
        return m
    except Exception as e:
        errs.append(("joblib", str(e)))
    try:
        import pickle
        with open(path, "rb") as f:
            m = pickle.load(f, encoding="latin1")
        return m
    except Exception as e:
        errs.append(("pickle_latin1", str(e)))
    try:
        import dill
        with open(path, "rb") as f:
            m = dill.load(f)
        return m
    except Exception as e:
        errs.append(("dill", str(e)))
    # If none worked raise an informative error
    raise RuntimeError("Model could not be loaded. Tried joblib, pickle(latin1), dill. Errors: " + repr(errs))


# Features in the model (from your CSV)
FEATURES = [
    "Number_of_Customers_Per_Day",
    "Average_Order_Value",
    "Operating_Hours_Per_Day",
    "Number_of_Employees",
    "Marketing_Spend_Per_Day",
    "Location_Foot_Traffic"
]

MODEL_PATH = os.path.join("models", "coffee.pkl")

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load model on startup (will raise error if not loadable)
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    # Keep a reference to the error so we can display it on the web UI
    model = None
    model_load_error = str(e)
else:
    model_load_error = None


@app.route("/", methods=["GET"])
def index():
    # Send feature names to template so form can be built dynamically
    return render_template("index.html", features=FEATURES, model_error=model_load_error)


def prepare_input(form):
    """
    Create a single-row pandas DataFrame with correct order and dtypes.
    Returns DataFrame of shape (1, len(FEATURES))
    """
    values = []
    for feat in FEATURES:
        raw = form.get(feat, "").strip()
        if raw == "":
            raise ValueError(f"Missing value for {feat}")
        # cast to float (allow integers)
        try:
            val = float(raw)
        except ValueError:
            raise ValueError(f"Invalid numeric value for {feat}: '{raw}'")
        values.append(val)
    # create DataFrame with column ordering matching FEATURES
    df = pd.DataFrame([values], columns=FEATURES)
    return df


@app.route("/predict", methods=["POST"])
def predict():
    global model, model_load_error
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded: " + (model_load_error or "unknown")}), 500

    try:
        df = prepare_input(request.form)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

    # prediction
    try:
        # Many scikit-learn models accept numpy arrays or DataFrames
        pred = model.predict(df)
        # assume single output
        revenue = float(pred[0])
    except Exception as e:
        return jsonify({"success": False, "error": "Model prediction failed: " + str(e)}), 500

    return jsonify({
        "success": True,
        "prediction": revenue,
        "input": df.to_dict(orient="records")[0]
    })


if __name__ == "__main__":
    # For local development
    app.run(debug=True, host="0.0.0.0", port=5000)
