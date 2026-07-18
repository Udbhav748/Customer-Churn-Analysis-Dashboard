"""Shared data loading/cleaning used by both training and the Streamlit app."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Customer_Data.csv"

NUMERIC_FEATURES = [
    "Age",
    "Number_of_Referrals",
    "Tenure_in_Months",
    "Monthly_Charge",
    "Total_Charges",
    "Total_Refunds",
    "Total_Extra_Data_Charges",
    "Total_Long_Distance_Charges",
    "Total_Revenue",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "State",
    "Value_Deal",
    "Phone_Service",
    "Multiple_Lines",
    "Internet_Service",
    "Internet_Type",
    "Online_Security",
    "Online_Backup",
    "Device_Protection_Plan",
    "Premium_Support",
    "Streaming_TV",
    "Streaming_Movies",
    "Streaming_Music",
    "Unlimited_Data",
    "Contract",
    "Paperless_Billing",
    "Payment_Method",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_training_data(path=DATA_PATH):
    """Features/target for churn prediction.

    Excludes 'Joined' customers: they have no churn history yet, so the
    label would be undefined for them.
    """
    df = load_raw(path)
    df = df[df["Customer_Status"] != "Joined"].copy()
    y = (df["Customer_Status"] == "Churned").astype(int)
    X = df[FEATURE_COLUMNS].copy()
    return X, y
