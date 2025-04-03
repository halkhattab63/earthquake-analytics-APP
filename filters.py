# filters.py
import pandas as pd

def apply_filters(df, mag_range, date_range, selected_province, selected_severity):
    """تطبيق فلاتر البيانات"""
    filtered_df = df[
        (df["magnitude"] >= mag_range[0]) & 
        (df["magnitude"] <= mag_range[1]) & 
        (df["time"] >= pd.to_datetime(date_range[0])) & 
        (df["time"] <= pd.to_datetime(date_range[1]))
    ]

    if selected_province != "Tümü":
        filtered_df = filtered_df[filtered_df["province"] == selected_province]

    if selected_severity != "Tümü":
        filtered_df = filtered_df[filtered_df["severity"] == selected_severity]

    return filtered_df
