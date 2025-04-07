# src/generate_report.py

import datetime
import os
import logging

import pandas as pd

# def generate_summary_report(output_path="outputs/reports/summary.md"):
#     logging.info("📝 Generating summary report...")
#     today = datetime.date.today().strftime("%Y-%m-%d")

#     content = f"""# 📊 Elazığ Earthquake Project Report
# **Date:** {today}

# ## Summary

# - ✅ **Earthquakes analyzed:** 10 years of data from USGS
# - ✅ **Shelters extracted from OSM:** clustered, mapped, and evaluated
# - ✅ **Risk levels classified:** Based on magnitude
# - ✅ **Evacuation paths:** Dijkstra algorithm
# - ✅ **Interactive dashboard and maps generated**

# ## Outputs

# - 🗺️ `outputs/maps/kde_heatmap.png`
# - 🗺️ `outputs/maps/evacuation_dashboard.html`
# - 🗺️ `outputs/maps/shelter_risk_map.html`
# - 📂 `outputs/tables/shelter_clusters.csv`
# - 📂 `outputs/tables/risk_levels.csv`

# ## Next Suggestions

# - Add real-time earthquake alerting
# - Connect population data to shelter capacity

# ---

# *This report was generated automatically.*
# """
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(content)

#     logging.info(f"✅ Summary report written to {output_path}")
def generate_summary_report(risk_csv_fp="outputs/tables/risk_levels.csv", out_md_fp="outputs/reports/risk_summary.md"):
    os.makedirs(os.path.dirname(out_md_fp), exist_ok=True)

    df = pd.read_csv(risk_csv_fp)

    total = len(df)
    low = (df.risk_level == "Low").sum()
    med = (df.risk_level == "Medium").sum()
    high = (df.risk_level == "High").sum()

    with open(out_md_fp, "w", encoding="utf-8") as f:
        f.write("# 📊 Earthquake Risk Summary Report\n\n")
        f.write(f"Total earthquakes: **{total}**\n\n")
        f.write(f"- 🟢 Low Risk: {low}\n")
        f.write(f"- 🟠 Medium Risk: {med}\n")
        f.write(f"- 🔴 High Risk: {high}\n\n")
        f.write("Generated automatically by the Elazığ Smart Pipeline.")

    logging.info(f"📝 Risk summary report written to {out_md_fp}")
