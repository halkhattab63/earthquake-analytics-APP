# src/preprocess.py

import geopandas as gpd
import pandas as pd
import os
import logging

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/preprocess.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def load_and_filter_earthquakes(file_path: str, min_magnitude: float = 4.0) -> gpd.GeoDataFrame:
    """
    Load earthquake shapefile and filter records by minimum magnitude.
    """
    logging.info(f"📂 Loading earthquake data from {file_path}")
    gdf = gpd.read_file(file_path)

    if 'mag' not in gdf.columns:
        msg = "❌ Column 'mag' is missing in the dataset."
        logging.error(msg)
        raise ValueError(msg)

    filtered = gdf[gdf['mag'] >= min_magnitude].copy()
    logging.info(f"🔍 Earthquakes filtered: {len(filtered)} / {len(gdf)}")
    return filtered

def prepare_for_kde(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Transform GeoDataFrame into flat DataFrame for KDE and risk analysis.
    Includes latitude, longitude, magnitude, and depth if available.
    """
    import logging

    required_cols = {"geometry", "mag"}
    missing = required_cols - set(gdf.columns)
    if missing:
        msg = f"❌ Missing required columns: {missing}"
        logging.error(msg)
        raise ValueError(msg)

    df = pd.DataFrame({
        "latitude": gdf.geometry.y,
        "longitude": gdf.geometry.x,
        "magnitude": gdf["mag"]
    })

    # Add depth if present
    if "depth" in gdf.columns:
        df["depth"] = gdf["depth"]
    else:
        df["depth"] = 10.0  # default fallback depth

    if df.isnull().any().any():
        logging.warning("⚠️ Missing values detected in prepared DataFrame.")

    logging.info(f"✅ Prepared DataFrame with {len(df)} rows, ready for KDE and risk classification.")
    return df


def save_cleaned(df: pd.DataFrame, out_path: str):
    """
    Save cleaned DataFrame as CSV.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logging.info(f"💾 Saved cleaned CSV to {out_path}")
