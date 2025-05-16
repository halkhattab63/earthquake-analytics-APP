# src/generate_report.py
import os
import logging
import pandas as pd
import plotly.express as px

def generate_summary_report(
    risk_csv_fp: str = "earthquake analytics APP/project yol/outputs/tables/risk_levels.csv",
    out_md_fp: str = "earthquake analytics APP/project yol/outputs/reports/risk_summary.md",
    chart_fp: str = "earthquake analytics APP/project yol/outputs/reports/risk_pie_chart.png"
) -> None:
    """
    Generate a Markdown summary report and pie chart of earthquake risk levels.

    Parameters:
        risk_csv_fp (str): Path to the input CSV file.
        out_md_fp (str): Path to the output Markdown file.
        chart_fp (str): Path to save the generated pie chart image.
    """
    try:
        df = pd.read_csv(risk_csv_fp)
    except FileNotFoundError:
        logging.error(f"CSV file not found: {risk_csv_fp}")
        return
    except pd.errors.EmptyDataError:
        logging.error(f"CSV file is empty: {risk_csv_fp}")
        return

    if "risk_level" not in df.columns:
        logging.error("Missing 'risk_level' column in CSV.")
        return

    os.makedirs(os.path.dirname(out_md_fp), exist_ok=True)

    total = len(df)
    risk_counts = df["risk_level"].value_counts().to_dict()

    # تحضير بيانات الرسم بناءً على القيم الفعلية فقط
    pie_df = pd.DataFrame([
        {"Risk Level": level, "Count": count}
        for level, count in risk_counts.items() if count > 0
    ])

    fig = px.pie(
        pie_df,
        names="Risk Level",
        values="Count",
        color="Risk Level",
        color_discrete_map={
            "Low": "green",
            "Medium": "orange",
            "High": "red",
            "Critical": "darkred"
        },
        title="Earthquake Risk Distribution"
    )
    fig.write_image(chart_fp)

    # توليد تقرير Markdown
    lines = [
        "# \U0001F4CA Earthquake Risk Summary Report\n",
        f"Total earthquakes: **{total}**\n",
    ]

    for level in ["Low", "Medium", "High", "Critical"]:
        if level in risk_counts:
            symbol = "🟢" if level == "Low" else "🟠" if level == "Medium" else "🔴" if level == "High" else "⚫"
            lines.append(f"- {symbol} {level} Risk: {risk_counts[level]}")

    lines += [
        f"\n![Risk Pie Chart]({os.path.basename(chart_fp)})",
        "\nGenerated automatically by the Elazığ Smart Pipeline.\n"
    ]

    with open(out_md_fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logging.info(f"✅ Summary written to {out_md_fp}")
    logging.info(f"📈 Chart saved to {chart_fp}")