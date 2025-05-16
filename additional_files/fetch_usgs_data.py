# src/fetch_usgs_data.py

import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from datetime import datetime, timedelta
import logging
import os

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Initialize logger
logging.basicConfig(filename="D:/books/4.1 donem/Bitirme projesi/codes/earthquake analytics APP/earthquake_risk_predictor/usgsdata/logs/usgs_fetch.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_usgs_earthquake_data(start_date, end_date, bbox):
    logging.info("📡 Fetching earthquake data from USGS...")
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_date.strftime("%Y-%m-%d"),
        "endtime": end_date.strftime("%Y-%m-%d"),
        "minlatitude": bbox[1],
        "maxlatitude": bbox[3],
        "minlongitude": bbox[0],
        "maxlongitude": bbox[2]
    }
    response = requests.get(url, params=params)
    data = response.json()

    features = data.get("features", [])
# src/fetch_usgs_data.py (جزء من الكود المسؤول عن تحويل JSON إلى DataFrame)

    records = []
    for f in data["features"]:
        props = f["properties"]
        coords = f["geometry"]["coordinates"]
        records.append({
            "time": datetime.utcfromtimestamp(props["time"] / 1000.0),
            "place": props["place"],
            "mag": props["mag"],  # ✅ تعديل الاسم هنا
            "depth": coords[2],
            "longitude": coords[0],
            "latitude": coords[1]
        })

    gdf = gpd.GeoDataFrame(records, geometry=gpd.points_from_xy([r["longitude"] for r in records], [r["latitude"] for r in records]), crs="EPSG:4326")

    return gdf


def save_earthquake_data(gdf, out_path):
    gdf.to_file(f"{out_path}.geojson", driver="GeoJSON")
    gdf.to_file(f"{out_path}.shp")
    gdf.drop(columns="geometry").to_csv(f"{out_path}.csv", index=False)
    logging.info(f"💾 Earthquake data saved to {out_path}.[geojson/shp/csv]")
