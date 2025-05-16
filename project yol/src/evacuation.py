# src/evacuation.py

import os
import math
import logging
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def filter_road_types(roads_gdf, allowed_types=None):
    if allowed_types is None:
        allowed_types = ["residential", "primary", "secondary", "tertiary"]

    if "highway" not in roads_gdf.columns:
        logging.warning("Column 'highway' not found. No filtering applied.")
        return roads_gdf

    filtered = roads_gdf[roads_gdf["highway"].isin(allowed_types)].copy()
    logging.info(f"Filtered {len(filtered)} of {len(roads_gdf)} roads based on allowed types.")
    return filtered

def build_road_graph(roads_gdf, max_segment_length=0.01):
    G = nx.Graph()
    skipped = 0

    for _, row in roads_gdf.iterrows():
        if isinstance(row.geometry, LineString):
            coords = list(row.geometry.coords)
            for i in range(len(coords) - 1):
                src, dst = coords[i], coords[i + 1]
                segment = LineString([src, dst])
                distance = segment.length

                if distance > max_segment_length:
                    skipped += 1
                    continue

                G.add_edge(src, dst, weight=distance)

    logging.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    logging.info(f"Skipped {skipped} long segments (>{max_segment_length} degrees).")
    return G

def get_nearest_node(G, point):
    return min(G.nodes, key=lambda n: math.dist(point, n))

def find_shortest_path(G, source_point, target_point):
    logging.info("Computing shortest path between given points.")
    try:
        src_node = get_nearest_node(G, source_point)
        tgt_node = get_nearest_node(G, target_point)
        path = nx.shortest_path(G, source=src_node, target=tgt_node, weight="weight")
        logging.info(f"Path found with {len(path)} nodes.")
        return path
    except nx.NetworkXNoPath:
        logging.error("❌ No path found between source and target points.")
        return []
    except Exception as e:
        logging.error(f"Unexpected error during pathfinding: {e}")
        return []

def path_to_geodataframe(G, path_nodes):
    if not path_nodes or len(path_nodes) < 2:
        logging.warning("⚠️ Path is empty or too short to convert to GeoDataFrame.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    segments = [LineString([path_nodes[i], path_nodes[i + 1]]) for i in range(len(path_nodes) - 1)]
    return gpd.GeoDataFrame(geometry=segments, crs="EPSG:4326")

def export_path(gdf_path, name):
    output_dir = "outputs/shapefiles"
    os.makedirs(output_dir, exist_ok=True)

    shapefile_path = os.path.join(output_dir, f"{name}.shp")
    geojson_path = os.path.join(output_dir, f"{name}.geojson")

    gdf_path.to_file(shapefile_path)
    gdf_path.to_file(geojson_path, driver="GeoJSON")

    logging.info(f"Evacuation path exported: {shapefile_path}, {geojson_path}")

def calculate_evacuation_path(roads_gdf, start_point, end_point):
    filtered = filter_road_types(roads_gdf)
    graph = build_road_graph(filtered, max_segment_length=0.01)
    path_nodes = find_shortest_path(graph, start_point, end_point)
    return path_to_geodataframe(graph, path_nodes)
