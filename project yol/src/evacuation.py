# src/evacuation.py

import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString
import logging
import math
import os

def filter_road_types(roads_gdf, allowed_types=None):
    """
    Filter road segments based on highway types.
    """
    if allowed_types is None:
        allowed_types = ["residential", "primary", "secondary", "tertiary"]

    if "highway" not in roads_gdf.columns:
        logging.warning("⚠️ Column 'highway' not found. No filtering applied.")
        return roads_gdf

    filtered = roads_gdf[roads_gdf["highway"].isin(allowed_types)].copy()
    logging.info(f"🧹 Filtered roads: {len(filtered)} / {len(roads_gdf)} based on type.")
    return filtered


def build_road_graph(roads_gdf, max_segment_length=0.01):
    """
    Converts a GeoDataFrame of LineString roads to a NetworkX graph,
    skipping segments longer than a threshold to avoid memory overflow.
    """
    logging.info("🛣️ Building road graph from LineStrings...")
    G = nx.Graph()
    skipped = 0

    for _, row in roads_gdf.iterrows():
        if isinstance(row.geometry, LineString):
            coords = list(row.geometry.coords)
            for i in range(len(coords) - 1):
                src, dst = coords[i], coords[i + 1]
                distance = LineString([src, dst]).length
                if distance > max_segment_length:
                    skipped += 1
                    continue
                G.add_edge(src, dst, weight=distance)

    logging.info(f"✅ Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    logging.info(f"⚠️ Skipped {skipped} long segments (> {max_segment_length} degrees).")
    return G


def get_nearest_node(G, point):
    """
    Find the nearest node in the graph G to a (lat, lon) point.
    """
    min_dist = float("inf")
    nearest_node = None
    for node in G.nodes:
        dist = math.dist((point[0], point[1]), (node[0], node[1]))
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    return nearest_node


def find_shortest_path(G, source_point, target_point):
    """
    Compute the shortest path in the road graph between two lat/lon points.
    """
    logging.info("🚶 Calculating shortest path...")
    source_node = get_nearest_node(G, source_point)
    target_node = get_nearest_node(G, target_point)
    path = nx.shortest_path(G, source=source_node, target=target_node, weight="weight")
    logging.info(f"✅ Shortest path contains {len(path)} points.")
    return path


def path_to_geodataframe(G, path_nodes):
    """
    Convert a list of path nodes to a GeoDataFrame.
    """
    lines = []
    for i in range(len(path_nodes) - 1):
        lines.append(LineString([path_nodes[i], path_nodes[i + 1]]))
    return gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")


def export_path(gdf_path, name):
    """
    Save the evacuation path to both GeoJSON and Shapefile.
    """
    os.makedirs("outputs/shapefiles", exist_ok=True)
    gdf_path.to_file(f"outputs/shapefiles/{name}.shp")
    gdf_path.to_file(f"outputs/shapefiles/{name}.geojson", driver="GeoJSON")
    logging.info(f"💾 Path exported to outputs/shapefiles/{name}.shp / .geojson")


def calculate_evacuation_path(roads_gdf, start_point, end_point):
    """
    Wrapper to build the graph, compute path and return GeoDataFrame.
    """
    filtered_roads = filter_road_types(roads_gdf)
    G = build_road_graph(filtered_roads, max_segment_length=0.01)
    path_nodes = find_shortest_path(G, start_point, end_point)
    path_gdf = path_to_geodataframe(G, path_nodes)
    return path_gdf
