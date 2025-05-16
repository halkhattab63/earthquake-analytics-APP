# src/visualize_risk_map.py

import os
import logging
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster

# Logging config
logging.basicConfig(
    filename="earthquake analytics APP/project yol/logs/visualize_risk_map.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def generate_risk_map(
    geojson_path: str = "earthquake analytics APP/project yol/outputs/tables/shelter_risk_joined.geojson",
    out_html: str = "earthquake analytics APP/project yol/outputs/maps/shelter_risk_map.html"
) -> None:
    """
    Generate a marker map with risk-level colored icons.
    """
    logging.info("🗺️ Generating colored risk map with markers...")

    gdf = gpd.read_file(geojson_path)
    fmap = folium.Map(location=[38.67, 39.22], zoom_start=13)
    cluster = MarkerCluster().add_to(fmap)

    color_map = {
        "Low": "green",
        "Medium": "orange",
        "High": "red",
        "Critical": "darkred"
    }

    for _, row in gdf.iterrows():
        risk = row.get("risk_level", "Unknown")
        color = color_map.get(risk, "gray")

        popup = folium.Popup(f"<b>Risk:</b> {risk}", max_width=200)
        marker = folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=popup,
            icon=folium.Icon(color=color, icon="info-sign")
        )
        marker.add_to(cluster)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fmap.save(out_html)
    logging.info(f"✅ Risk map saved to {out_html}")

def generate_risk_colored_map(
    gdf_fp: str = "earthquake analytics APP/project yol/outputs/tables/shelter_risk_joined.geojson",
    out_html: str = "earthquake analytics APP/project yol/outputs/maps/shelter_risk_map.html"
) -> None:
    """
    Generate a circle marker map colored by risk levels.
    """
    logging.info("🌐 Generating circle marker risk map...")

    gdf = gpd.read_file(gdf_fp)
    gdf = gdf.to_crs(epsg=4326)
    gdf = gdf[gdf.geometry.notnull() & (gdf.geometry.geom_type == "Point")]

    fmap = folium.Map(location=[38.67, 39.22], zoom_start=13)

    color_map = {
        "Low": "green",
        "Medium": "orange",
        "High": "red",
        "Critical": "darkred"
    }

    for _, row in gdf.iterrows():
        risk = row.get("risk_level", "Unknown")
        color = color_map.get(risk, "gray")

        popup = folium.Popup(f"<b>Risk:</b> {risk}", max_width=200)
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup
        ).add_to(fmap)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fmap.save(out_html)
    logging.info(f"✅ Circle risk map saved to {out_html}")