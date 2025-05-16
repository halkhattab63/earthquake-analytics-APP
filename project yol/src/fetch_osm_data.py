import os
import osmnx as ox
import geopandas as gpd

def fetch_road_network(place_name: str) -> gpd.GeoDataFrame:
    """Download the drivable road network from OpenStreetMap for a specified place."""
    print(f"🚦 Downloading road network for: {place_name}")
    graph = ox.graph_from_place(place_name, network_type="drive")
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)

    if edges.empty:
        print(f"⚠️ No road segments found for {place_name}.")
    else:
        print(f"✅ Retrieved {len(edges)} road segments.")
    
    return edges

def fetch_shelter_pois(place_name: str) -> gpd.GeoDataFrame:
    """Download potential shelter POIs from OpenStreetMap and return them as centroids."""
    print(f"🏕️ Downloading potential shelter POIs for: {place_name}")
    tags = {
        "amenity": ["shelter", "school", "hospital"],
        "building": ["public", "civic", "school"],
        "leisure": ["park"]
    }

    pois = ox.features_from_place(place_name, tags)

    required_cols = ['geometry', 'name', 'amenity', 'building', 'leisure']
    pois = pois[[col for col in required_cols if col in pois.columns]].copy()

    if pois.empty:
        print("⚠️ No POIs found for this place.")
        return pois

    pois = pois.to_crs(epsg=3857)
    pois["geometry"] = pois.geometry.centroid
    pois = pois.to_crs(epsg=4326)
    pois = gpd.GeoDataFrame(pois, geometry="geometry", crs="EPSG:4326")

    print(f"✅ Retrieved {len(pois)} POIs (converted to centroids).")
    return pois

def save_geodataframe(gdf: gpd.GeoDataFrame, output_path: str):
    """Save GeoDataFrame to Shapefile and GeoJSON format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    geom_cols = gdf.select_dtypes(include="geometry").columns.tolist()
    if not geom_cols:
        print("⚠️ No geometry column found in the GeoDataFrame.")
        return

    main_geom = geom_cols[0]
    gdf = gdf.drop(columns=[col for col in geom_cols if col != main_geom])
    if main_geom != "geometry":
        gdf = gdf.rename(columns={main_geom: "geometry"})

    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    gdf.to_file(f"{output_path}.shp")
    gdf.to_file(f"{output_path}.geojson", driver="GeoJSON")

    print(f"💾 Saved to: {output_path}.shp / .geojson")
