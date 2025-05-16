import os
import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopandas.tools import sjoin_nearest

# Logging configuration
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "risk_join.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def classify_risk(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Classify seismic risk levels based on the 'magnitude' column.

    Parameters:
        input_csv (str): Path to input CSV containing 'magnitude'.
        output_csv (str): Path to save output CSV with 'risk_level'.

    Returns:
        pd.DataFrame: DataFrame with assigned risk levels.
    """
    logging.info("🔎 Classifying seismic risk levels...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_csv}")
        raise

    if "magnitude" not in df.columns:
        raise ValueError("Input CSV must contain a 'magnitude' column.")

    bins = [0, 4.5, 5.5, 6.5, 10]
    labels = ['Low', 'Medium', 'High', 'Critical']

    df["risk_level"] = pd.cut(df["magnitude"], bins=bins, labels=labels, include_lowest=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logging.info(f"✅ Risk levels saved to {output_csv}")
    return df


def join_risk_with_shelters(
    risk_fp: str = "earthquake analytics APP/project yol/outputs/tables/risk_levels.csv",
    shelters_fp: str = "earthquake analytics APP/project yol/data/shapefiles/elazig_shelters.shp",
    out_fp: str = "earthquake analytics APP/project yol/outputs/tables/shelter_risk_joined.geojson"
) -> None:
    """
    Join shelter locations with the nearest risk points and assign custom icons.

    Parameters:
        risk_fp (str): Path to CSV file with classified risk data (must include lat/lon).
        shelters_fp (str): Path to shelter shapefile (must contain geometries).
        out_fp (str): Path to output joined GeoJSON.
    """
    logging.info("🔗 Starting spatial join with risk classification...")

    try:
        gdf_shelters = gpd.read_file(shelters_fp)
        df_risk = pd.read_csv(risk_fp)
    except Exception as e:
        logging.error(f"Error loading input files: {e}")
        raise

    if "longitude" not in df_risk.columns or "latitude" not in df_risk.columns:
        raise ValueError("Risk CSV must contain 'longitude' and 'latitude' columns.")

    gdf_risk = gpd.GeoDataFrame(
        df_risk,
        geometry=gpd.points_from_xy(df_risk.longitude, df_risk.latitude),
        crs="EPSG:4326"
    )

    gdf_shelters = gdf_shelters.to_crs(epsg=3857)
    gdf_risk = gdf_risk.to_crs(epsg=3857)

    joined = sjoin_nearest(gdf_shelters, gdf_risk, how="left")
    logging.info(f"Joined {len(joined)} shelters with nearest risk points.")
    joined = joined.to_crs(epsg=4326)

    joined["type"] = joined["type"].fillna("shelter").str.lower() if "type" in joined.columns else "shelter"
    joined["image"] = joined["image"].fillna("https://cdn-icons-png.flaticon.com/512/684/684908.png") if "image" in joined.columns else "https://cdn-icons-png.flaticon.com/512/684/684908.png"
    joined["description"] = joined["description"].fillna("Identified as an emergency shelter option during earthquakes.") if "description" in joined.columns else "Identified as an emergency shelter option during earthquakes."

    icon_map = {
        "hospital": "https://cdn-icons-png.flaticon.com/512/4521/4521422.png",
        "school": "https://cdn-icons-png.flaticon.com/512/5310/5310672.png",
        "shelter": "https://cdn-icons-png.flaticon.com/512/11797/11797639.png",
        "park": "https://cdn1.iconfinder.com/data/icons/map-objects/154/map-object-tree-park-forest-point-place-512.png",
        "tent": "https://cdn-icons-png.flaticon.com/512/4605/4605575.png"
    }

    joined["icon_url"] = joined["type"].map(icon_map).fillna(icon_map["shelter"])

    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    joined.to_file(out_fp, driver="GeoJSON")
    logging.info(f"✅ Output saved: {out_fp}")
