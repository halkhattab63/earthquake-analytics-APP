# src/visualize_risk_map.py

import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import os
import logging

def generate_risk_map(geojson_path="outputs/tables/shelter_risk_joined.geojson", out_html="outputs/maps/shelter_risk_map.html"):
    logging.info("🗺️ Generating colored risk map...")
    gdf = gpd.read_file(geojson_path)

    fmap = folium.Map(location=[38.67, 39.22], zoom_start=13)
    cluster = MarkerCluster().add_to(fmap)

    risk_colors = {
        "Low": "green",
        "Medium": "orange",
        "High": "red",
        "Critical": "darkred"
    }

    for _, row in gdf.iterrows():
        risk = row.get("risk_level", "Unknown")
        color = risk_colors.get(risk, "gray")
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=f"{risk} risk",
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(cluster)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fmap.save(out_html)
    logging.info(f"✅ Risk map saved to {out_html}")
    
def generate_risk_colored_map(gdf_fp="outputs/tables/shelter_risk_joined.geojson", out_html="outputs/maps/shelter_risk_map.html"):
    gdf = gpd.read_file(gdf_fp)

    gdf = gdf.to_crs(epsg=4326)
    gdf = gdf[gdf.geometry.notnull() & (gdf.geometry.type == "Point")]

    fmap = folium.Map(location=[38.67, 39.22], zoom_start=13)

    color_map = {
        "Low": "green",
        "Moderate": "orange",
        "High": "red",
    }

    for _, row in gdf.iterrows():
        risk = row.get("risk_level", "Unknown")
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=6,
            color=color_map.get(risk, "gray"),
            fill=True,
            fill_color=color_map.get(risk, "gray"),
            fill_opacity=0.7,
            popup=f"{risk} risk"
        ).add_to(fmap)

    fmap.save(out_html)
    print(f"🗺️ Risk-colored shelter map saved to {out_html}")