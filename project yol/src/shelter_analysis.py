import os
import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
import numpy as np

# Logging setup
log_path = "earthquake analytics APP/project yol/logs/shelter_analysis.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def cluster_shelters(gdf: gpd.GeoDataFrame, n_clusters: int = 5) -> tuple[gpd.GeoDataFrame, list]:
    """
    Apply KMeans clustering to shelter coordinates safely.

    Parameters:
        gdf: GeoDataFrame with Point geometries
        n_clusters: requested number of clusters

    Returns:
        gdf with 'cluster' column and list of center coordinates
    """
    if gdf.empty or gdf.geometry.is_empty.all():
        raise ValueError("Input GeoDataFrame is empty or lacks valid geometries.")

    coords = gdf.geometry.apply(lambda p: [p.y, p.x]).tolist()
    num_samples = len(coords)

    if num_samples < 2:
        logging.warning("⚠️ Not enough shelters to cluster. Assigning all to cluster 0.")
        gdf["cluster"] = 0
        return gdf, []

    if n_clusters > num_samples:
        logging.warning(f"⚠️ Reducing clusters from {n_clusters} to {num_samples}")
        n_clusters = num_samples

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    gdf["cluster"] = kmeans.fit_predict(coords)
    centers = kmeans.cluster_centers_

    logging.info(f"✅ Clustering completed with {n_clusters} clusters.")
    return gdf, centers

def save_clustered_data(gdf: gpd.GeoDataFrame, centers: list, output_path: str) -> None:
    """
    Save clustered GeoDataFrame and cluster centers to disk.

    Parameters:
        gdf: Clustered GeoDataFrame
        centers: Cluster centers (lat, lon)
        output_path: Base path without extension
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    gdf.to_csv(f"{output_path}.csv", index=False)
    gdf.to_file(f"{output_path}.shp")
    gdf.to_file(f"{output_path}.geojson", driver="GeoJSON")
    logging.info(f"💾 Clustered data saved to {output_path}.*")

    if isinstance(centers, (list, np.ndarray)) and len(centers) > 0:
        df_centers = pd.DataFrame(centers, columns=["latitude", "longitude"])
        df_centers.to_csv(f"{output_path}_centers.csv", index=False)
        logging.info(f"📌 Cluster centers saved to {output_path}_centers.csv")
