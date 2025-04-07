# # project/main.py

# import sys

# from src.generate_report import generate_summary_report
# from src.visualize_risk_map import generate_risk_colored_map, generate_risk_map
# sys.stdout.reconfigure(encoding='utf-8')

# import os
# import geopandas as gpd
# from shapely.geometry import Point
# from datetime import datetime, timedelta

# from src.fetch_usgs_data import fetch_usgs_earthquake_data, save_earthquake_data
# from src.preprocess import load_and_filter_earthquakes, prepare_for_kde, save_cleaned
# from src.analyze_kde import perform_kde, plot_kde_heatmap
# from src.fetch_osm_data import fetch_road_network, fetch_shelter_pois, save_geodataframe
# from src.generate_maps import generate_combined_paths_map, generate_interactive_map
# from src.shelter_analysis import cluster_shelters, save_clustered_data
# from src.evacuation import (
#     build_road_graph,
#     find_shortest_path,
#     path_to_geodataframe,
#     export_path,
#     calculate_evacuation_path
# )
# from src.risk_classifier import classify_risk, join_risk_with_shelters
# from src.dashboard import build_dashboard, generate_risk_map_html

# import logging

# os.makedirs("logs", exist_ok=True)
# logging.basicConfig(filename="logs/main.log", level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


# def run_pipeline():
#     logging.info("🚀 Starting full pipeline...")

#     # 1. Earthquake data
#     end = datetime.utcnow()
#     start = end - timedelta(days=365 * 10)
#     bbox = [37.0, 38.0, 40.0, 39.5]

#     gdf = fetch_usgs_earthquake_data(start, end, bbox)
#     save_earthquake_data(gdf, "data/raw/earthquakes_turkey")

#     # 2. Preprocessing
#     filtered_gdf = load_and_filter_earthquakes("data/raw/earthquakes_turkey.shp", min_magnitude=4.0)
#     clean_df = prepare_for_kde(filtered_gdf)
#     save_cleaned(clean_df, "data/cleaned/earthquakes_cleaned.csv")

#     # 3. KDE Heatmap
#     xx, yy, density = perform_kde(clean_df)
#     plot_kde_heatmap(xx, yy, density, "outputs/maps/kde_heatmap.png")

#     # 4. Road and Shelter data from OSM
#     roads = fetch_road_network("Elazığ, Turkey")
#     save_geodataframe(roads, "data/shapefiles/elazig_roads")

#     shelters = fetch_shelter_pois("Elazığ, Turkey")
#     save_geodataframe(shelters, "data/shapefiles/elazig_shelters")

#     # 5. Build road graph
#     roads = gpd.read_file("data/shapefiles/elazig_roads.shp")
#     G = build_road_graph(roads)

#     # 6. Main evacuation path
#     main_start = (38.675, 39.221)
#     main_end = (38.685, 39.235)
#     path_nodes = find_shortest_path(G, main_start, main_end)
#     path_gdf = path_to_geodataframe(G, path_nodes)
#     export_path(path_gdf, "evacuation_path")

#     # 7. Web Map
#     generate_interactive_map(
#     roads_fp="data/shapefiles/elazig_roads.shp",
#     shelters_fp="outputs/tables/shelter_risk_joined.geojson",  # ✅ هنا التعديل
#     path_fp="outputs/shapefiles/evacuation_path.shp",
#     output_html="outputs/maps/evacuation_dashboard.html"
# )
#     generate_combined_paths_map(
#         "data/shapefiles/elazig_roads.shp",
#         "data/shapefiles/elazig_shelters.shp",
#         [f"outputs/shapefiles/evacuation_path_{i+1}.geojson" for i in range(3)],
#         "outputs/maps/multi_paths_map.html"
#     )

#     # 8. Shelter clustering
#     clustered_shelters, centers = cluster_shelters(shelters, n_clusters=5)
#     save_clustered_data(clustered_shelters, centers, "outputs/tables/shelter_clusters")

#     # 9. Multiple evacuation paths
#     sample_points = [Point(39.218, 38.676), Point(39.225, 38.682), Point(39.212, 38.670)]
#     for i, start_point in enumerate(sample_points):
#         shelter_point = shelters.geometry.iloc[i % len(shelters)]
#         path_gdf = calculate_evacuation_path(roads, (start_point.y, start_point.x), (shelter_point.y, shelter_point.x))
#         export_path(path_gdf, f"evacuation_path_{i+1}")

#     # 10. Risk Classification
#     classify_risk("data/cleaned/earthquakes_cleaned.csv", "outputs/tables/risk_levels.csv")

#     logging.info("✅ All steps completed.")




#     # 11. Risk-shelter join
# # # ✅ الصحيح بدون مسافة في البداية
# #     join_risk_with_shelters(
# #         risk_csv="outputs/tables/risk_levels.csv",
# #         shelters_fp="data/shapefiles/elazig_shelters.shp",
# #         out_fp="outputs/tables/shelter_risk_joined.geojson"
# #     )

#     join_risk_with_shelters()
#     generate_summary_report()
#     generate_risk_map_html()
#     # 10. Risk-colored map
#     generate_risk_map()
#     generate_risk_colored_map()
#     # 11. Summary report
#     # generate_summary_report("outputs/tables/risk_levels.csv", "outputs/reports/risk_summary.md")

# if __name__ == "__main__":
#     run_pipeline()
#     app = build_dashboard()
#     app.run(debug=True, use_reloader=False)




# project/main.py

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import logging
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta

from src.fetch_usgs_data import fetch_usgs_earthquake_data, save_earthquake_data
from src.preprocess import load_and_filter_earthquakes, prepare_for_kde, save_cleaned
from src.analyze_kde import perform_kde, plot_kde_heatmap
from src.fetch_osm_data import fetch_road_network, fetch_shelter_pois, save_geodataframe
from src.generate_maps import generate_combined_paths_map, generate_interactive_map
from src.shelter_analysis import cluster_shelters, save_clustered_data
from src.evacuation import (
    build_road_graph, filter_road_types, find_shortest_path,
    path_to_geodataframe, export_path,
    calculate_evacuation_path
)
from src.risk_classifier import classify_risk, join_risk_with_shelters
from src.generate_report import generate_summary_report
from src.visualize_risk_map import generate_risk_colored_map, generate_risk_map
from src.dashboard import build_dashboard, generate_risk_map_html
# Logging setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/main.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
def run_pipeline():
    logging.info("🚀 Starting full pipeline...")

    # 1. Earthquake data
    from datetime import datetime, timedelta, timezone
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * 10)
    bbox = [37.0, 38.0, 40.0, 39.5]

    gdf = fetch_usgs_earthquake_data(start, end, bbox)
    save_earthquake_data(gdf, "data/raw/earthquakes_turkey")

    # 2. Preprocessing
    filtered_gdf = load_and_filter_earthquakes("data/raw/earthquakes_turkey.shp", min_magnitude=4.0)
    clean_df = prepare_for_kde(filtered_gdf)
    save_cleaned(clean_df, "data/cleaned/earthquakes_cleaned.csv")

    if clean_df.empty:
        logging.error("❌ Empty DataFrame from prepare_for_kde. Pipeline halted.")
        return

    # 3. KDE analysis
    xx, yy, density = perform_kde(clean_df)
    plot_kde_heatmap(xx, yy, density, "outputs/maps/kde_heatmap.png")


    # 4. Risk zone polygons


    # 5. OSM data
    place_name = "Elazığ, Turkey"
    roads = fetch_road_network(place_name)
    save_geodataframe(roads, "data/shapefiles/elazig_roads")

    shelters = fetch_shelter_pois(place_name)
    save_geodataframe(shelters, "data/shapefiles/elazig_shelters")

    # 6. Risk classification
    classify_risk("data/cleaned/earthquakes_cleaned.csv", "outputs/tables/risk_levels.csv")

    join_risk_with_shelters(
        risk_fp="outputs/tables/risk_levels.csv",
        shelters_fp="data/shapefiles/elazig_shelters.shp",
        out_fp="outputs/tables/shelter_risk_joined.geojson"
    )

    # 7. Intersect risk zones with shelters and roads
    # intersect_with_risk_zones(
    #     risk_zones_fp="outputs/shapefiles/risk_zones.geojson",
    #     shelters_fp="outputs/tables/shelter_risk_joined.geojson",
    #     roads_fp="data/shapefiles/elazig_roads.shp",
    #     out_shelters_fp="outputs/shapefiles/risky_shelters.geojson",
    #     out_roads_fp="outputs/shapefiles/risky_roads.geojson"
    # )

    # visualize_risk_zones_and_shelters(
    #     risk_fp="outputs/shapefiles/risk_zones.geojson",
    #     shelters_fp="outputs/shapefiles/risky_shelters.geojson",
    #     roads_fp="outputs/shapefiles/risky_roads.geojson",
    #     out_html="outputs/maps/risk_zones_map.html"
    # )

    # 8. Evacuation path
    roads = gpd.read_file("data/shapefiles/elazig_roads.shp")
    filtered_roads = filter_road_types(roads)
    G = build_road_graph(filtered_roads, max_segment_length=0.01)
    main_start = (38.675, 39.221)
    main_end = (38.685, 39.235)
    path_nodes = find_shortest_path(G, main_start, main_end)
    main_path = path_to_geodataframe(G, path_nodes)
    export_path(main_path, "evacuation_path")

    # 9. Multiple paths to shelters
    shelters_gdf = gpd.read_file("outputs/tables/shelter_risk_joined.geojson")
    sample_points = [Point(39.218, 38.676), Point(39.225, 38.682), Point(39.212, 38.670)]
    for i, start_point in enumerate(sample_points):
        shelter_point = shelters_gdf.geometry.iloc[i % len(shelters_gdf)]
        path_gdf = calculate_evacuation_path(roads, (start_point.y, start_point.x), (shelter_point.y, shelter_point.x))
        export_path(path_gdf, f"evacuation_path_{i+1}")

    # 10. Clustering
    clustered_shelters, centers = cluster_shelters(shelters, n_clusters=5)
    save_clustered_data(clustered_shelters, centers, "outputs/tables/shelter_clusters")

    # 11. Maps
    generate_interactive_map(
        roads_fp="data/shapefiles/elazig_roads.shp",
        shelters_fp="outputs/tables/shelter_risk_joined.geojson",
        path_fp="outputs/shapefiles/evacuation_path.shp",
        output_html="outputs/maps/evacuation_dashboard.html"
    )

    generate_combined_paths_map(
        "data/shapefiles/elazig_roads.shp",
        "outputs/tables/shelter_risk_joined.geojson",
        [f"outputs/shapefiles/evacuation_path_{i+1}.geojson" for i in range(3)],
        "outputs/maps/multi_paths_map.html"
    )

    # 12. Reports
    generate_summary_report()
    generate_risk_map_html()
    generate_risk_map()
    generate_risk_colored_map()

    logging.info("✅ Full pipeline completed successfully.")


# def run_pipeline():
#     logging.info("🚀 Starting full pipeline...")

#     # 1. Earthquake data
#     end = datetime.utcnow()
#     start = end - timedelta(days=365 * 10)
#     bbox = [37.0, 38.0, 40.0, 39.5]
#     gdf = fetch_usgs_earthquake_data(start, end, bbox)
#     save_earthquake_data(gdf, "data/raw/earthquakes_turkey")

#     # 2. Preprocessing
#     filtered_gdf = load_and_filter_earthquakes("data/raw/earthquakes_turkey.shp", min_magnitude=4.0)
#     clean_df = prepare_for_kde(filtered_gdf)
#     save_cleaned(clean_df, "data/cleaned/earthquakes_cleaned.csv")




# # -----------------------------------------
#     # 4. Extract and visualize high-density risk zones
#     extract_high_density_zones(
#         xx, yy, density,
#         threshold=0.9,
#         output_path="outputs/shapefiles/risk_zones"
#     )

#     intersect_with_risk_zones(
#         risk_zones_fp="outputs/shapefiles/risk_zones.geojson",
#         shelters_fp="outputs/tables/shelter_risk_joined.geojson",
#         roads_fp="data/shapefiles/elazig_roads.shp",
#         out_shelters_fp="outputs/shapefiles/risky_shelters.geojson",
#         out_roads_fp="outputs/shapefiles/risky_roads.geojson"
#     )

#     visualize_risk_zones_and_shelters(
#         risk_fp="outputs/shapefiles/risk_zones.geojson",
#         shelters_fp="outputs/shapefiles/risky_shelters.geojson",
#         roads_fp="outputs/shapefiles/risky_roads.geojson",
#         out_html="outputs/maps/risk_zones_map.html"
#     )


# # -----------------------------------------------


#     # 3. KDE analysis
#     xx, yy, density = perform_kde(clean_df)
#     plot_kde_heatmap(xx, yy, density, "outputs/maps/kde_heatmap.png")
#     # اسم المنطقة المطلوبة
#     place_name = "Elazığ, Turkey"

#     # تحميل شبكة الطرق
#     roads_gdf = fetch_road_network(place_name)
    
#     # تحميل نقاط الاهتمام للملاجئ
#     shelters_gdf = fetch_shelter_pois(place_name)
    
#     # حفظ البيانات في الملفات
#     save_geodataframe(roads_gdf, "data/shapefiles/elazig_roads")
#     save_geodataframe(shelters_gdf, "data/shapefiles/elazig_shelters")
#     # 4. Fetch road + shelter data
#     roads = fetch_road_network("Elazığ, Turkey")
#     save_geodataframe(roads, "data/shapefiles/elazig_roads")
#     shelters = fetch_shelter_pois("Elazığ, Turkey")
#     save_geodataframe(shelters, "data/shapefiles/elazig_shelters")

#     # 5. Risk Classification
#     classify_risk("data/cleaned/earthquakes_cleaned.csv", "outputs/tables/risk_levels.csv")

#     # 6. Join shelters with risk
#     join_risk_with_shelters(
#         risk_fp="outputs/tables/risk_levels.csv",
#         shelters_fp="data/shapefiles/elazig_shelters.shp",
#         out_fp="outputs/tables/shelter_risk_joined.geojson"
#     )

#     # 7. Road Graph & main path
#     roads = gpd.read_file("data/shapefiles/elazig_roads.shp")
#     G = build_road_graph(roads)
#     main_start = (38.675, 39.221)
#     main_end = (38.685, 39.235)
#     path_nodes = find_shortest_path(G, main_start, main_end)
#     main_path = path_to_geodataframe(G, path_nodes)
#     export_path(main_path, "evacuation_path")

#     # 8. Multiple evacuation paths
#     shelters_gdf = gpd.read_file("outputs/tables/shelter_risk_joined.geojson")
#     sample_points = [Point(39.218, 38.676), Point(39.225, 38.682), Point(39.212, 38.670)]
#     for i, start_point in enumerate(sample_points):
#         shelter_point = shelters_gdf.geometry.iloc[i % len(shelters_gdf)]
#         path_gdf = calculate_evacuation_path(roads, (start_point.y, start_point.x), (shelter_point.y, shelter_point.x))
#         export_path(path_gdf, f"evacuation_path_{i+1}")

#     # 9. Cluster shelters
#     clustered_shelters, centers = cluster_shelters(shelters, n_clusters=5)
#     save_clustered_data(clustered_shelters, centers, "outputs/tables/shelter_clusters")

#     # 10. Generate maps using joined shelter data
#     generate_interactive_map(
#         roads_fp="data/shapefiles/elazig_roads.shp",
#         shelters_fp="outputs/tables/shelter_risk_joined.geojson",  # ✅ now with risk_level
#         path_fp="outputs/shapefiles/evacuation_path.shp",
#         output_html="outputs/maps/evacuation_dashboard.html"
#     )
#     generate_combined_paths_map(
#         "data/shapefiles/elazig_roads.shp",
#         "outputs/tables/shelter_risk_joined.geojson",
#         [f"outputs/shapefiles/evacuation_path_{i+1}.geojson" for i in range(3)],
#         "outputs/maps/multi_paths_map.html"
#     )

#     # 11. Reports and visualizations
#     generate_summary_report()
#     generate_risk_map_html()
#     generate_risk_map()
#     generate_risk_colored_map()

#     logging.info("✅ Full pipeline completed successfully.")


if __name__ == "__main__":
    
    run_pipeline()
    app = build_dashboard()
    app.run(debug=True, use_reloader=False)
