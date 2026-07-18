import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from ml.data import CATEGORICAL_FEATURES, DATA_PATH, FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

BLUE = "#2a78d6"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
GRID = "#e1e0d9"
CHART_FONT = "system-ui, -apple-system, Segoe UI, sans-serif"

st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📉", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e1e0d9;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text())


def style_chart(fig, title, horizontal=False, height=340):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#0b0b0b")),
        font=dict(family=CHART_FONT, size=13, color="#52514e"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=height,
        showlegend=False,
    )
    value_axis = fig.update_xaxes if horizontal else fig.update_yaxes
    label_axis = fig.update_yaxes if horizontal else fig.update_xaxes
    value_axis(showgrid=True, gridcolor=GRID, zeroline=False, title="")
    label_axis(showgrid=False, zeroline=False, title="")
    fig.update_traces(marker_cornerradius=4, selector=dict(type="bar"))
    return fig


df = load_data()
model = load_model()
metrics = load_metrics()

st.sidebar.title("📉 Churn Dashboard")
st.sidebar.write(
    "Telecom customer churn analysis, built on the original Power BI dashboard "
    "plus a Random Forest model trained on the same data."
)
st.sidebar.metric("Model ROC-AUC", f"{metrics['results'][metrics['best_model']]['roc_auc']:.2f}")
st.sidebar.divider()
st.sidebar.markdown("[GitHub repo](https://github.com/Udbhav748/Customer-Churn-Analysis-Dashboard)")
st.sidebar.caption("Udbhav N · BTech Computer Science")

st.title("Customer Churn Dashboard")
st.caption("Segmentation from the Power BI dashboard, rebuilt in Python, plus live churn prediction.")

tab_overview, tab_predict, tab_batch = st.tabs(["Overview", "Predict", "Batch Predict"])

with tab_overview:
    total = len(df)
    joined = (df.Customer_Status == "Joined").sum()
    churned = (df.Customer_Status == "Churned").sum()
    churn_rate = churned / total

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("New Joiners", f"{joined:,}")
    c3.metric("Total Churn", f"{churned:,}")
    c4.metric("Churn Rate", f"{churn_rate:.1%}")

    st.write("")

    known = df[df.Customer_Status != "Joined"].copy()
    known["Churned"] = known.Customer_Status == "Churned"

    left, right = st.columns(2)

    with left, st.container(border=True):
        by_contract = (
            known.groupby("Contract")["Churned"].mean().sort_values(ascending=False).reset_index()
        )
        fig = px.bar(by_contract, x="Contract", y="Churned", color_discrete_sequence=[BLUE])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_chart(fig, "Churn Rate by Contract Type"), use_container_width=True)

    with right, st.container(border=True):
        by_internet = (
            known.groupby("Internet_Service")["Churned"].mean().sort_values(ascending=False).reset_index()
        )
        fig = px.bar(by_internet, x="Internet_Service", y="Churned", color_discrete_sequence=[BLUE])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_chart(fig, "Churn Rate by Internet Service"), use_container_width=True)

    left2, right2 = st.columns(2)

    with left2, st.container(border=True):
        by_payment = (
            known.groupby("Payment_Method")["Churned"].mean().sort_values(ascending=False).reset_index()
        )
        fig = px.bar(by_payment, x="Payment_Method", y="Churned", color_discrete_sequence=[BLUE])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_chart(fig, "Churn Rate by Payment Method"), use_container_width=True)

    with right2, st.container(border=True):
        reasons = known.loc[known.Churned, "Churn_Category"].value_counts().reset_index()
        reasons.columns = ["Churn_Category", "Count"]
        fig = px.bar(
            reasons, x="Count", y="Churn_Category", orientation="h", color_discrete_sequence=[CRITICAL]
        )
        st.plotly_chart(style_chart(fig, "Churn Reasons", horizontal=True), use_container_width=True)

    with st.container(border=True):
        by_state = (
            known.groupby("State")["Churned"].mean().sort_values(ascending=False).head(10).reset_index()
        )
        fig = px.bar(by_state, x="State", y="Churned", color_discrete_sequence=[BLUE])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(
            style_chart(fig, "Top 10 States by Churn Rate", height=380), use_container_width=True
        )

    with st.container(border=True):
        importances = pd.DataFrame(metrics["feature_importances"][:10]).sort_values("importance")
        fig = px.bar(
            importances, x="importance", y="feature", orientation="h", color_discrete_sequence=[BLUE]
        )
        st.plotly_chart(
            style_chart(fig, "Top Churn Drivers (model feature importance)", horizontal=True, height=380),
            use_container_width=True,
        )

with tab_predict:
    st.subheader("Predict churn risk for a customer")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 18, 100, 40)
            gender = st.selectbox("Gender", ["Female", "Male"])
            married = st.selectbox("Married", ["Yes", "No"])
            state = st.selectbox("State", sorted(df.State.dropna().unique()))
            tenure = st.number_input("Tenure (months)", 0, 100, 12)
            referrals = st.number_input("Number of Referrals", 0, 20, 0)

        with col2:
            contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
            payment = st.selectbox("Payment Method", ["Credit Card", "Bank Withdrawal", "Mailed Check"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            value_deal = st.selectbox(
                "Value Deal", ["No Deal", "Deal 1", "Deal 2", "Deal 3", "Deal 4", "Deal 5"]
            )
            monthly_charge = st.number_input("Monthly Charge", -50.0, 200.0, 65.0)

        with col3:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"], disabled=phone_service == "No")
            internet_service = st.selectbox("Internet Service", ["Yes", "No"])
            internet_type = st.selectbox(
                "Internet Type", ["Fiber Optic", "DSL", "Cable"], disabled=internet_service == "No"
            )

        if internet_service == "Yes":
            st.caption("Internet add-ons")
            addon_row1 = st.columns(4)
            online_security = addon_row1[0].selectbox("Online Security", ["No", "Yes"])
            online_backup = addon_row1[1].selectbox("Online Backup", ["No", "Yes"])
            device_protection = addon_row1[2].selectbox("Device Protection", ["No", "Yes"])
            premium_support = addon_row1[3].selectbox("Premium Support", ["No", "Yes"])
            addon_row2 = st.columns(4)
            streaming_tv = addon_row2[0].selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = addon_row2[1].selectbox("Streaming Movies", ["No", "Yes"])
            streaming_music = addon_row2[2].selectbox("Streaming Music", ["No", "Yes"])
            unlimited_data = addon_row2[3].selectbox("Unlimited Data", ["No", "Yes"])
        else:
            internet_type = "No Service"
            online_security = online_backup = device_protection = premium_support = "No Service"
            streaming_tv = streaming_movies = streaming_music = unlimited_data = "No Service"

        if phone_service == "No":
            multiple_lines = "No Service"

        st.caption("Billing history")
        bill_cols = st.columns(4)
        total_charges = bill_cols[0].number_input(
            "Total Charges", 0.0, 20000.0, float(monthly_charge * tenure)
        )
        total_refunds = bill_cols[1].number_input("Total Refunds", 0.0, 500.0, 0.0)
        extra_data_charges = bill_cols[2].number_input("Total Extra Data Charges", 0, 1000, 0)
        long_distance = bill_cols[3].number_input("Total Long Distance Charges", 0.0, 5000.0, 0.0)
        total_revenue = total_charges - total_refunds + extra_data_charges + long_distance

        predict_clicked = st.button("Predict", type="primary")

    if predict_clicked:
        row = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Number_of_Referrals": referrals,
                    "Tenure_in_Months": tenure,
                    "Monthly_Charge": monthly_charge,
                    "Total_Charges": total_charges,
                    "Total_Refunds": total_refunds,
                    "Total_Extra_Data_Charges": extra_data_charges,
                    "Total_Long_Distance_Charges": long_distance,
                    "Total_Revenue": total_revenue,
                    "Gender": gender,
                    "Married": married,
                    "State": state,
                    "Value_Deal": "No Service" if value_deal == "No Deal" else value_deal,
                    "Phone_Service": phone_service,
                    "Multiple_Lines": multiple_lines,
                    "Internet_Service": internet_service,
                    "Internet_Type": internet_type,
                    "Online_Security": online_security,
                    "Online_Backup": online_backup,
                    "Device_Protection_Plan": device_protection,
                    "Premium_Support": premium_support,
                    "Streaming_TV": streaming_tv,
                    "Streaming_Movies": streaming_movies,
                    "Streaming_Music": streaming_music,
                    "Unlimited_Data": unlimited_data,
                    "Contract": contract,
                    "Paperless_Billing": paperless,
                    "Payment_Method": payment,
                }
            ]
        )[FEATURE_COLUMNS]

        proba = model.predict_proba(row)[0, 1]
        risk = "Low" if proba < 0.33 else "Medium" if proba < 0.66 else "High"

        with st.container(border=True):
            result_col, factor_col = st.columns([1, 2])

            with result_col:
                st.metric("Churn Probability", f"{proba:.0%}")
                if risk == "Low":
                    st.success(f"{risk} risk")
                elif risk == "Medium":
                    st.warning(f"{risk} risk")
                else:
                    st.error(f"{risk} risk")

            with factor_col:
                row_values = row.iloc[0].to_dict()
                matches = []
                for item in metrics["feature_importances"]:
                    feat = item["feature"]
                    for col in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
                        prefix = col + "_"
                        if feat.startswith(prefix) and str(row_values.get(col)) == feat[len(prefix):]:
                            matches.append((col.replace("_", " "), feat[len(prefix):]))
                            break

                if matches:
                    st.caption("Contributing factors for this customer")
                    for col, val in matches[:5]:
                        st.write(f"• **{col}:** {val}")

with tab_batch:
    st.subheader("Batch churn prediction")

    with st.container(border=True):
        st.write("Upload a CSV with the same columns as `Customer_Data.csv`.")
        file = st.file_uploader("CSV file", type="csv")

        if file is not None:
            batch = pd.read_csv(file)
            missing = [c for c in FEATURE_COLUMNS if c not in batch.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                proba = model.predict_proba(batch[FEATURE_COLUMNS])[:, 1]
                result = batch.copy()
                result["Churn_Probability"] = proba
                result["Risk_Level"] = pd.cut(
                    proba, [-0.01, 0.33, 0.66, 1.0], labels=["Low", "Medium", "High"]
                )
                st.dataframe(result)
                st.download_button(
                    "Download predictions",
                    result.to_csv(index=False).encode("utf-8"),
                    "churn_predictions.csv",
                    "text/csv",
                )
