import gzip
import json
import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table
from sklearn.metrics import confusion_matrix, classification_report

# ==============================
# Paths 
# ==============================
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "poland_bankruptcy-data-2009.json.gz"
MODEL_PATH = BASE_DIR / "final_model.pkl"
ARTIFACTS_PATH = BASE_DIR / "dashboard_artifacts.pkl"

# ==============================
# Load and wrangle data
# ==============================

def wrangle(json_gz_path):
    
    with gzip.open(json_gz_path, "rt", encoding="utf-8") as f:
        poland_data = json.load(f)

    df = pd.DataFrame(poland_data)

    # Rename Attr1..Attr64 -> feat_1..feat_64, class -> bankrupt
    rename_map = {f"Attr{i}": f"feat_{i}" for i in range(1, 65)}
    rename_map["class"] = "bankrupt"
    df = df.rename(columns=rename_map)

    return df


df = wrangle(DATA_PATH)

# ==============================
# Load trained model and artifacts
# ==============================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ARTIFACTS_PATH, "rb") as f:
    artifacts = pickle.load(f)

X_test = artifacts["X_test"]
y_test = artifacts["y_test"]
fpr = artifacts["fpr"]
tpr = artifacts["tpr"]
auc_roc = artifacts["auc_roc"]
feat_imp = artifacts["feat_imp"]

# Ensure numeric labels
y_test = pd.Series(y_test).astype(int)
y_test_pred = model.predict(X_test).astype(int)

# All features used by the trained model
ALL_FEATURES = list(model.feature_names_in_)

# Top-10 features to expose in the UI
TOP_FEATURES = [
    "feat_27",
    "feat_34",
    "feat_24",
    "feat_5",
    "feat_46",
    "feat_26",
    "feat_6",
    "feat_16",
    "feat_13",
    "feat_35",
]

# ==============================
# Classification report
# ==============================

report_dict = classification_report(
    y_test,
    y_test_pred,
    target_names=["Not bankrupt", "Bankrupt"],
    output_dict=True,
    digits=4,
)

report_df = (
    pd.DataFrame(report_dict)
    .transpose()
    .reset_index()
    .rename(columns={"index": "Class"})
    .round(4)
)

# ==============================
# EDA figures
# ==============================

# Histogram
fig_target = px.histogram(
    df,
    x="bankrupt",
    title="Target Distribution (Bankrupt)",
)

# Donut chart (class balance)
class_counts = df["bankrupt"].value_counts(normalize=True).reset_index()
class_counts.columns = ["bankrupt", "frequency"]

fig_pie = px.pie(
    class_counts,
    names="bankrupt",
    values="frequency",
    title="Class Balance",
    hole=0.4,
)

# Correlation matrix
fig_corr = px.imshow(
    df[TOP_FEATURES + ["bankrupt"]].corr(),
    color_continuous_scale="RdBu_r",
    title="Correlation: Top Features vs Target",
    aspect="auto",
)

fig_corr.update_layout(width=1000, height=800)
fig_corr.update_xaxes(tickangle=45)

# ==============================
# Metrics figures
# ==============================

cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

fig_cm = px.imshow(
    cm,
    x=["Not Bankrupt (0)", "Bankrupt (1)"],
    y=["Not Bankrupt (0)", "Bankrupt (1)"],
    text_auto=True,
    color_continuous_scale="Blues",
    labels=dict(x="Predicted", y="Actual", color="Count"),
    title="Confusion Matrix – Test Set",
)

fig_roc = go.Figure()
fig_roc.add_trace(
    go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc_roc:.3f})")
)
fig_roc.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Baseline",
        line=dict(dash="dash"),
    )
)
fig_roc.update_layout(
    title="ROC Curve – Test Set",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
)

# ==============================
# Feature importance (top 10)
# ==============================

top10 = feat_imp.sort_values().tail(10).reset_index()
top10.columns = ["feature", "importance"]

fig_feat_imp = px.bar(
    top10,
    x="importance",
    y="feature",
    orientation="h",
    title="Top 10 Feature Importances – Tuned Gradient Boosting",
)
fig_feat_imp.update_layout(yaxis={"categoryorder": "total ascending"})

# ==============================
# Dash app
# ==============================

app = Dash(__name__)
server = app.server 

app.layout = html.Div(
    [
        html.H1("Bankruptcy Prediction Dashboard (Poland)"),

        dcc.Tabs(
            value="tab-eda",
            children=[
                dcc.Tab(
                    label="EDA",
                    children=[
                        html.H3("Exploratory Data Analysis"),
                        html.Div(
                            [
                                dcc.Graph(figure=fig_target, style={"width": "48%"}),
                                dcc.Graph(figure=fig_pie, style={"width": "48%"}),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                            },
                        ),
                        dcc.Graph(figure=fig_corr),
                    ],
                ),
                dcc.Tab(
                    label="Metrics",
                    children=[
                        html.H3("Model Performance"),
                        dcc.Graph(figure=fig_cm),
                        dcc.Graph(figure=fig_roc),
                    ],
                ),
                dcc.Tab(
                    label="Model",
                    children=[
                        html.H3("Feature Importance"),
                        dcc.Graph(figure=fig_feat_imp),

                        html.H3("Classification Report (Test Set)"),
                        dash_table.DataTable(
                            data=report_df.to_dict("records"),
                            columns=[{"name": c, "id": c} for c in report_df.columns],
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "textAlign": "center",
                                "padding": "8px",
                                "fontSize": "14px",
                            },
                            style_header={
                                "backgroundColor": "#f0f0f0",
                                "fontWeight": "bold",
                            },
                        ),

                        html.H3("Predict Bankruptcy (Top Features Only)"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(feat),
                                        dcc.Input(
                                            id=f"input-{feat}",
                                            type="number",
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    style={"marginBottom": "10px"},
                                )
                                for feat in TOP_FEATURES
                            ],
                            style={"columnCount": 2},
                        ),

                        html.Button("Predict", id="predict-button"),
                        html.Div(
                            id="prediction-output",
                            style={"marginTop": "20px"},
                        ),
                    ],
                ),
            ],
        ),
    ]
)

# ==============================
# Prediction callback
# ==============================

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    [State(f"input-{feat}", "value") for feat in TOP_FEATURES],
    prevent_initial_call=True,
)
def predict_top_features(n_clicks, *values):
    if any(v is None for v in values):
        return "Please fill in all feature values."

    X_new = pd.DataFrame(0, index=[0], columns=ALL_FEATURES)
    X_new[TOP_FEATURES] = values

    proba = model.predict_proba(X_new)[0, 1]
    pred = model.predict(X_new)[0]

    label = "Bankrupt" if pred == 1 else "Not Bankrupt"

    return f"Prediction: {label} | Probability of bankruptcy = {proba:.3f}"


if __name__ == "__main__":
    app.run(debug=True)
