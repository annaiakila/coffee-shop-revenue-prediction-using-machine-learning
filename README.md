**Project Overview**

This repository contains a small Flask web app that predicts daily coffee-shop revenue using a pre-trained machine-learning model. The app exposes a minimal web UI (forms) and a `/predict` endpoint that accepts form data and returns a JSON response with the predicted revenue.

**Contents**
- **`app.py`**: Flask application and prediction API.
- **`models/coffee.pkl`**: Serialized trained model used for prediction (loaded with `joblib`, `pickle` or `dill`).
- **`templates/index.html`**: Frontend HTML template that renders the form and shows results.
- **`static/css/styles.css`**: Frontend stylesheet used by the HTML template.
- **`coffeeshoprevenue.csv`**: (Optional) dataset used to train or inspect features.
- **`requirements.txt`**: Python dependencies.

**Project File Structure (recommended view)**

```
coffee/
├─ app.py
├─ requirements.txt
├─ coffeeshoprevenue.csv
├─ models/
│  └─ coffee.pkl
├─ templates/
│  └─ index.html
└─ static/
   └─ css/
      └─ styles.css
```

**Quick Setup (Windows / PowerShell)**

1. Create a virtual environment and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app locally:

```powershell
# option A: run with the module entry
python app.py
# option B: use flask CLI (if you prefer)
set FLASK_APP=app.py; flask run
```

Open `http://localhost:5000` in your browser.

**How the Frontend Works**
- The HTML template `templates/index.html` dynamically builds the form fields from the `FEATURES` list provided by the server.
- Styles are loaded from `static/css/styles.css`. If you change CSS and don't see updates in the browser, force a full reload (Ctrl+F5) or add a cache-busting query string in the template, for example:

```html
<link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}?v=1.1">
```

**Backend / API**
- Endpoint: `POST /predict`
  - Accepts form-encoded values (the web form uses `application/x-www-form-urlencoded` via `fetch` FormData).
  - Required fields (names must match exactly):

```
Number_of_Customers_Per_Day
Average_Order_Value
Operating_Hours_Per_Day
Number_of_Employees
Marketing_Spend_Per_Day
Location_Foot_Traffic
```

- Example `curl` (form data) to call the API directly:

```powershell
curl -X POST "http://127.0.0.1:5000/predict" -F Number_of_Customers_Per_Day=120 -F Average_Order_Value=4.5 -F Operating_Hours_Per_Day=10 -F Number_of_Employees=4 -F Marketing_Spend_Per_Day=25 -F Location_Foot_Traffic=300
```

- Example expected JSON response:

```json
{
  "success": true,
  "prediction": 2345.67,
  "input": {
    "Number_of_Customers_Per_Day": 120.0,
    "Average_Order_Value": 4.5,
    "Operating_Hours_Per_Day": 10.0,
    "Number_of_Employees": 4.0,
    "Marketing_Spend_Per_Day": 25.0,
    "Location_Foot_Traffic": 300.0
  }
}
```

**Model details and maintenance**

- **Model file**: `models/coffee.pkl`
  - `app.py` attempts to load the model using `joblib.load`, then `pickle.load(..., encoding='latin1')`, then `dill.load`. If your model was saved with one method, ensure the file was created using a compatible serializer.

- **Input expectations**:
  - The model is called with a single-row `pandas.DataFrame` containing the six features listed above in the exact order specified in `app.py` (the `FEATURES` list). Most scikit-learn models accept a DataFrame or numpy array.

- **To replace the model**:
  1. Retrain locally (example using scikit-learn):

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

df = pd.read_csv('coffeeshoprevenue.csv')
# assume df has the six feature columns and a target column named 'Revenue'
X = df[[
    'Number_of_Customers_Per_Day', 'Average_Order_Value', 'Operating_Hours_Per_Day',
    'Number_of_Employees', 'Marketing_Spend_Per_Day', 'Location_Foot_Traffic'
]]
y = df['Revenue']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
dump(model, 'models/coffee.pkl')
```

  2. Place the new `coffee.pkl` in the `models/` folder and restart the Flask server. The app attempts to load the model on startup; if loading fails the error is shown on the UI.

**Error handling & troubleshooting**

- If the UI shows a `Model Load Error`, check the terminal for the detailed exception; common causes:
  - Serializer mismatch (try saving with `joblib.dump(model, 'models/coffee.pkl')`).
  - Missing dependency (install from `requirements.txt`).

- If the UI loads but `/predict` returns a 400 error, ensure all fields are provided and numeric.

- If CSS does not appear:
  - Confirm `templates/index.html` references `css/styles.css` (not `style.css`).
  - Clear browser cache or add a version query param as shown earlier.

**Development notes**

- The server runs with `debug=True` in `app.py` for development. For production, use a WSGI server (e.g., Gunicorn) and disable debug mode.
- To expose an API for external clients, consider adding authentication and input validation (e.g., `flask-wtf` or `pydantic`).

**Suggested next steps**

- Add unit tests for model input validation and for the `/predict` endpoint using `pytest` and Flask's test client.
- Add a simple Dockerfile for containerized deployment.
- Add a small script to validate and print model feature importance (if model supports it).

**License & attribution**

This README is provided as guidance; include your preferred license if you plan to publish the project.

---
Generated by the project maintainer assistant. If you'd like, I can:
- Add a `Dockerfile` and `docker-compose.yml` to run the app in a container.
- Add a short `CONTRIBUTING.md` and basic tests.

**Notebook: `coffee.ipynb`**

This repository also includes a Jupyter notebook, `coffee.ipynb`, for exploratory data analysis and preprocessing. The notebook is a step-by-step EDA and preprocessing guide you can run locally to inspect the dataset and prepare features for modeling.

Highlights of `coffee.ipynb`:
- Loads `coffeeshoprevenue.csv` and shows quick inspections (`head()`, `info()`, `describe()`, missing-value checks).
- Univariate analysis (histograms, boxplots) and categorical countplots.
- Bivariate and trivariate visualizations: scatterplots, pairplots, 3D scatter, grouped lineplots, heatmaps, and stacked counts to explore interactions and correlations.
- Correlation heatmaps and outlier detection/removal (IQR method) for `Daily_Revenue`.
- Preprocessing examples: feature selection into `X`, target `Y`, scaling with `StandardScaler`, and a train/test split using `train_test_split`.

Notebook dependencies (install if needed):

```powershell
pip install jupyterlab notebook matplotlib seaborn scipy
```

Run the notebook locally:

```powershell
# activate your venv if using one
.\.venv\Scripts\Activate.ps1
# start Jupyter Lab or Notebook in the project directory
jupyter lab
# or
jupyter notebook
```

Tips:
- For faster iteration on large datasets, sample the data first (e.g., `df.sample(frac=0.2, random_state=42)`).
- Run cells in order; some later cells depend on variables defined earlier (e.g., `numeric_cols`).
- The notebook focuses on EDA and preprocessing; to persist a trained model use `joblib.dump(model, 'models/coffee.pkl')` and place it in the `models/` folder.

If you'd like, I can convert the notebook into a runnable script or add a training cell that writes `models/coffee.pkl` automatically.
