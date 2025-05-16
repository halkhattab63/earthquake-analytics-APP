import os
import logging
import geopandas as gpd
import pandas as pd

# Setup logging
LOG_FILE = "earthquake analytics APP/project yol/logs/preprocess.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_and_filter_earthquakes(file_path: str, min_magnitude: float = 4.0) -> gpd.GeoDataFrame:
    """
    Load earthquake shapefile and filter records by minimum magnitude.

    Parameters:
        file_path (str): Path to shapefile
        min_magnitude (float): Threshold magnitude for filtering

    Returns:
        GeoDataFrame: Filtered earthquakes
    """
    logging.info(f"📂 Loading earthquake data from {file_path}")
    
    gdf = gpd.read_file(file_path)

    if 'mag' not in gdf.columns:
        msg = "❌ Column 'mag' is missing in the dataset."
        logging.error(msg)
        raise ValueError(msg)

    filtered = gdf[gdf['mag'] >= min_magnitude].copy()
    logging.info(f"🔍 Filtered {len(filtered)} of {len(gdf)} earthquakes (≥ {min_magnitude})")
    return filtered

def prepare_for_kde(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Transform GeoDataFrame into flat DataFrame for KDE and risk analysis.
    
    Parameters:
        gdf (GeoDataFrame): Earthquake points with magnitude
    
    Returns:
        DataFrame: Cleaned data with lat/lon, magnitude, and optional depth
    """
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

    df["depth"] = gdf["depth"] if "depth" in gdf.columns else 10.0

    if df.isnull().any().any():
        logging.warning("⚠️ Missing values detected in the prepared DataFrame.")

    logging.info(f"✅ Prepared DataFrame with {len(df)} rows for KDE analysis.")
    return df

def save_cleaned(df: pd.DataFrame, out_path: str) -> None:
    """
    Save cleaned DataFrame as CSV.

    Parameters:
        df (DataFrame): Cleaned data
        out_path (str): Path to save CSV
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logging.info(f"💾 Saved cleaned CSV to {out_path}")
