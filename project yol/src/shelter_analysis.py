import os
import logging
import geopandas as gpd
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(filename="logs/shelter_analysis.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def cluster_shelters(gdf, n_clusters=5):
    logging.info(f"📊 Clustering {len(gdf)} shelters into {n_clusters} groups...")
    coords = gdf.geometry.apply(lambda p: [p.y, p.x]).tolist()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    gdf["cluster"] = kmeans.fit_predict(coords)
    centers = kmeans.cluster_centers_
    logging.info("✅ Clustering complete.")
    return gdf, centers


def save_clustered_data(gdf, centers, output_path):
    csv_path = f"{output_path}.csv"
    shp_path = f"{output_path}.shp"
    geojson_path = f"{output_path}.geojson"

    gdf.to_csv(csv_path, index=False)
    gdf.to_file(shp_path)
    gdf.to_file(geojson_path, driver="GeoJSON")

    logging.info(f"💾 Clustered shelters saved to: {csv_path}, {shp_path}, {geojson_path}")

    # Optional: Save cluster centers to a separate CSV
    df_centers = pd.DataFrame(centers, columns=["latitude", "longitude"])
    df_centers.to_csv(f"{output_path}_centers.csv", index=False)
    logging.info(f"📌 Cluster centers saved to {output_path}_centers.csv")
