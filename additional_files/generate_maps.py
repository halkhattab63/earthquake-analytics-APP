
import os
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from folium import FeatureGroup, LayerControl
from shapely.strtree import STRtree
import logging

os.makedirs("logs", exist_ok=True)

logging.basicConfig(filename="logs/map_generation.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def assign_risk_to_roads(roads_gdf, quakes_df, buffer_dist=0.001):
    """
    تصنيف الطرق حسب قربها من الزلازل إلى High/Medium/Low risk.
    """
    logging.info("🚦 Classifying road segments based on earthquake proximity...")

    # التأكد من CRS الصحيح
    if roads_gdf.crs is None:
        roads_gdf.set_crs("EPSG:4326", inplace=True)
    if roads_gdf.crs.to_string() != "EPSG:4326":
        roads_gdf = roads_gdf.to_crs("EPSG:4326")

    quake_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(quakes_df.longitude, quakes_df.latitude), crs="EPSG:4326")
    quake_tree = STRtree(quake_points.geometry)
    quake_geoms = quake_points.geometry.tolist()

    risk_levels = []

    for geom in roads_gdf.geometry:
        matches_idx = quake_tree.query(geom.buffer(buffer_dist))
        nearby_quakes = [quake_geoms[i] for i in matches_idx if quake_geoms[i].distance(geom) <= buffer_dist]

        count = len(nearby_quakes)
        if count >= 5:
            risk_levels.append("High")
        elif count >= 2:
            risk_levels.append("Medium")
        else:
            risk_levels.append("Low")

    # تحديث العمود
    roads_gdf["risk_level"] = risk_levels
    logging.info("✅ Risk assigned to all roads.")
    return roads_gdf


def generate_interactive_map(
    roads_fp,
    shelters_fp,
    path_fp,
    output_html,
    quakes_fp="data/cleaned/earthquakes_cleaned.csv"
):
    logging.info("🗺️ Generating enhanced interactive map...")

    roads = gpd.read_file(roads_fp)
    shelters = gpd.read_file(shelters_fp)
    path = gpd.read_file(path_fp)
    quakes = pd.read_csv(quakes_fp)

    roads = assign_risk_to_roads(roads, quakes)

    center = [shelters.geometry.y.mean(), shelters.geometry.x.mean()]
    fmap = folium.Map(location=center, zoom_start=13)

    # الطبقات حسب مستوى الخطر
    for level, color in [("Low", "green"), ("Medium", "orange"), ("High", "red")]:
        group = FeatureGroup(name=f"Road Risk: {level}")
        subset = roads[roads["risk_level"] == level]

    
        for level, color in [("Low", "green"), ("Medium", "orange"), ("High", "red")]:
            group = FeatureGroup(name=f"Road Risk: {level}")
            subset = roads[roads.get("risk_level") == level]

            if not subset.empty and "risk_level" in subset.columns:
                folium.GeoJson(
                    subset,
                    style_function=lambda x, col=color: {"color": col, "weight": 2},
                    tooltip=folium.GeoJsonTooltip(fields=["risk_level"], aliases=["Risk Level"])
                ).add_to(group)
            else:
                folium.GeoJson(
                    subset,
                    style_function=lambda x, col=color: {"color": col, "weight": 2}
                ).add_to(group)

            fmap.add_child(group)


    # الملاجئ
    shelter_cluster = MarkerCluster(name="Shelters").add_to(fmap)
    for _, row in shelters.iterrows():
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            icon=folium.Icon(color="blue", icon="home"),
            popup=row.get("type", "Shelter")
        ).add_to(shelter_cluster)
    park_cluster = MarkerCluster(name="Parks").add_to(fmap)
    for _, row in shelters.iterrows():
        location_type = str(row.get("type", "")).lower()
        if "park" in location_type:
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                icon=folium.Icon(color="blue", icon="leaf"),
                popup="Park"
            ).add_to(park_cluster)
    # مسار الإخلاء الآمن
    if not path.empty:
        path = path.to_crs("EPSG:4326")
        folium.GeoJson(
            path,
            name="Evacuation Route",
            style_function=lambda x: {"color": "lime", "weight": 5},
            tooltip="✅ Safe Evacuation Route"
        ).add_to(fmap)
    else:
        logging.warning("⚠️ Evacuation path is empty.")

    fmap.add_child(LayerControl())
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fmap.save(output_html)
    logging.info(f"✅ Map saved to {output_html}")

def generate_combined_paths_map(roads_fp, shelters_fp, path_files, output_html):
    logging.info("🗺️ Generating multi-path evacuation map...")

    roads = gpd.read_file(roads_fp)
    shelters = gpd.read_file(shelters_fp)

    fmap = folium.Map(location=[shelters.geometry.y.mean(), shelters.geometry.x.mean()], zoom_start=13)

    folium.GeoJson(
        roads,
        name="Road Network",
        style_function=lambda x: {"color": "gray", "weight": 1}
    ).add_to(fmap)

    shelter_cluster = MarkerCluster(name="Shelters").add_to(fmap)
    for _, row in shelters.iterrows():
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            icon=folium.Icon(color="blue", icon="home"),
            popup=row.get("type", "Shelter")
        ).add_to(shelter_cluster)

    for i, path_fp in enumerate(path_files):
        path = gpd.read_file(path_fp)
        path = path.to_crs("EPSG:4326")
        folium.GeoJson(
            path,
            name=f"Evacuation Route {i+1}",
            style_function=lambda x, color="green" if i == 0 else "blue": {"color": color, "weight": 4},
            tooltip=f"Path {i+1}"
        ).add_to(fmap)

    fmap.add_child(LayerControl())
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fmap.save(output_html)
    logging.info(f"✅ Multi-path map saved to {output_html}")


