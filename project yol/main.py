
# # project/main.py

# import sys
# sys.stdout.reconfigure(encoding='utf-8')

# import os
# import logging
# import geopandas as gpd
# from shapely.geometry import Point
# from datetime import datetime, timedelta

# from src.preprocess import load_and_filter_earthquakes, prepare_for_kde, save_cleaned

# from src.fetch_osm_data import fetch_road_network, fetch_shelter_pois, save_geodataframe
# from src.generate_maps import generate_combined_paths_map, generate_interactive_map
# from src.shelter_analysis import cluster_shelters, save_clustered_data
# from src.evacuation import (
#     build_road_graph, filter_road_types, find_shortest_path,
#     path_to_geodataframe, export_path,
#     calculate_evacuation_path
# )
# from src.risk_classifier import classify_risk, join_risk_with_shelters
# from src.generate_report import generate_summary_report
# from src.visualize_risk_map import generate_risk_colored_map, generate_risk_map
# from src.dashboard import build_dashboard, generate_risk_map_html
# # Logging setup
# os.makedirs("logs", exist_ok=True)
# logging.basicConfig(filename="earthquake_analytics_APP/project_yol/logs/main.log", level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# def run_pipeline():
#     logging.info("🚀 Starting full pipeline...")

#     # 2. Preprocessing
#     filtered_gdf = load_and_filter_earthquakes("earthquake_analytics_APP/project_yol/data/raw/earthquakes_turkey.shp", min_magnitude=4.0)
#     clean_df = prepare_for_kde(filtered_gdf)
#     save_cleaned(clean_df, "earthquake_analytics_APP/project_yol/data/cleaned/earthquakes_cleaned.csv")

#     if clean_df.empty:
#         logging.error("❌ Empty DataFrame from prepare_for_kde. Pipeline halted.")
#         return


#     # 5. OSM data
#     place_name = "Elazığ, Turkey"
#     roads = fetch_road_network(place_name)
#     save_geodataframe(roads, "earthquake_analytics_APP/project_yol/data/shapefiles/elazig_roads")

#     shelters = fetch_shelter_pois(place_name)
#     save_geodataframe(shelters, "earthquake_analytics_APP/project_yol/data/shapefiles/elazig_shelters")

#     # 6. Risk classification
#     classify_risk("earthquake_analytics_APP/project_yol/data/cleaned/earthquakes_cleaned.csv", "earthquake_analytics_APP/project_yol/outputs/tables/risk_levels.csv")

#     join_risk_with_shelters(
#         risk_fp="earthquake_analytics_APP/project_yol/outputs/tables/risk_levels.csv",
#         shelters_fp="earthquake_analytics_APP/project_yol/data/shapefiles/elazig_shelters.shp",
#         out_fp="earthquake_analytics_APP/project_yol/outputs/tables/shelter_risk_joined.geojson"
#     )

#     # 8. Evacuation path
#     roads = gpd.read_file("earthquake_analytics_APP/project_yol/data/shapefiles/elazig_roads.shp")
#     filtered_roads = filter_road_types(roads)
#     G = build_road_graph(filtered_roads, max_segment_length=0.01)
#     main_start = (38.675, 39.221)
#     main_end = (38.685, 39.235)
#     path_nodes = find_shortest_path(G, main_start, main_end)
#     main_path = path_to_geodataframe(G, path_nodes)
#     export_path(main_path, "evacuation_path")

#     # 9. Multiple paths to shelters
#     shelters_gdf = gpd.read_file("earthquake_analytics_APP/project_yol/outputs/tables/shelter_risk_joined.geojson")
#     sample_points = [Point(39.218, 38.676), Point(39.225, 38.682), Point(39.212, 38.670)]
#     for i, start_point in enumerate(sample_points):
#         shelter_point = shelters_gdf.geometry.iloc[i % len(shelters_gdf)]
#         path_gdf = calculate_evacuation_path(roads, (start_point.y, start_point.x), (shelter_point.y, shelter_point.x))
#         export_path(path_gdf, f"evacuation_path_{i+1}")

#     # 10. Clustering
#     clustered_shelters, centers = cluster_shelters(shelters, n_clusters=5)
#     save_clustered_data(clustered_shelters, centers, "earthquake_analytics_APP/project_yol/outputs/tables/shelter_clusters")

#     # 11. Maps
#     generate_interactive_map(
#         roads_fp="earthquake_analytics_APP/project_yol/data/shapefiles/elazig_roads.shp",
#         shelters_fp="earthquake_analytics_APP/project_yol/outputs/tables/shelter_risk_joined.geojson",
#         path_fp="earthquake_analytics_APP/project_yol/outputs/shapefiles/evacuation_path.shp",
#         output_html="earthquake_analytics_APP/project_yol/outputs/maps/evacuation_dashboard.html"
#     )

#     generate_combined_paths_map(
#         "earthquake_analytics_APP/project_yol/data/shapefiles/elazig_roads.shp",
#         "earthquake_analytics_APP/project_yol/outputs/tables/shelter_risk_joined.geojson",
#         [f"earthquake_analytics_APP/project_yol/outputs/shapefiles/evacuation_path_{i+1}.geojson" for i in range(3)],
#         "earthquake_analytics_APP/project_yol/outputs/maps/multi_paths_map.html"
#     )

#     # 12. Reports
#     generate_summary_report()
#     generate_risk_map_html()
#     generate_risk_map()
#     generate_risk_colored_map()

#     logging.info("✅ Full pipeline completed successfully.")


# if __name__ == "__main__":
    
#     run_pipeline()
#     app = build_dashboard()
#     app.run(debug=True, use_reloader=False)



# main.py

# src/main.py

import os
import logging
from src.fetch_osm_data import fetch_road_network, fetch_shelter_pois, save_geodataframe
from src.risk_classifier import classify_risk, join_risk_with_shelters
from src.shelter_analysis import cluster_shelters, save_clustered_data
from src.generate_report import generate_summary_report
from src.visualize_risk_map import generate_risk_map
from src.dashboard import build_dashboard
import sys
sys.stdout.reconfigure(encoding='utf-8')

# إعداد المسارات
PLACE_NAME = "Elazığ, Turkey"
RAW_DATA_DIR = "earthquake analytics APP/project yol/data/raw"
CLEANED_DATA_DIR = "earthquake analytics APP/project yol/data/cleaned"
TABLES_DIR = "earthquake analytics APP/project yol/outputs/tables"
MAPS_DIR = "earthquake analytics APP/project yol/outputs/maps"
REPORTS_DIR = "earthquake analytics APP/project yol/outputs/reports"

os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(MAPS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ملفات الإدخال/الإخراج
EARTHQUAKE_CSV = os.path.join(CLEANED_DATA_DIR, "earthquakes_cleaned.csv")
RISK_CSV = os.path.join(TABLES_DIR, "risk_levels.csv")
JOINED_GEOJSON = os.path.join(TABLES_DIR, "shelter_risk_joined.geojson")
CLUSTER_OUTPUT_PATH = os.path.join(TABLES_DIR, "shelters_clustered")
SUMMARY_MD = os.path.join(REPORTS_DIR, "risk_summary.md")
SUMMARY_IMG = os.path.join(REPORTS_DIR, "risk_pie_chart.png")

def run_pipeline():
    logging.info("🚀 Running Earthquake Risk Analysis Pipeline")

    # 1. تحميل بيانات OpenStreetMap
    roads = fetch_road_network(PLACE_NAME)
    shelters = fetch_shelter_pois(PLACE_NAME)

    save_geodataframe(roads, os.path.join(CLEANED_DATA_DIR, "roads"))
    save_geodataframe(shelters, os.path.join(CLEANED_DATA_DIR, "shelters"))

    # ملاحظة: يجب أن يكون لديك ملف earthquakes_cleaned.csv في CLEANED_DATA_DIR مسبقًا

    # 2. تصنيف المخاطر
    classify_risk(EARTHQUAKE_CSV, RISK_CSV)

    # 3. ربط الملاجئ بمستوى المخاطر
    join_risk_with_shelters(RISK_CSV, f"{CLEANED_DATA_DIR}/shelters.shp", JOINED_GEOJSON)

    # 4. تجميع الملاجئ وتحليلها
    gdf_shelters = shelters
    gdf_shelters_clustered, centers = cluster_shelters(gdf_shelters)
    save_clustered_data(gdf_shelters_clustered, centers, CLUSTER_OUTPUT_PATH)

    # 5. توليد خريطة ملونة
    generate_risk_map(JOINED_GEOJSON, f"{MAPS_DIR}/shelter_risk_map.html")

    # 6. توليد تقرير
    generate_summary_report(RISK_CSV, SUMMARY_MD, SUMMARY_IMG)

    # 7. تشغيل لوحة البيانات
    app = build_dashboard()
    app.run_server(debug=True)

if __name__ == "__main__":
    run_pipeline()
