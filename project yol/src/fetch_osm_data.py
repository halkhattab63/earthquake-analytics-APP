# # src/fetch_osm_data.py

# import os
# import osmnx as ox
# import geopandas as gpd

# def fetch_road_network(place_name: str) -> gpd.GeoDataFrame:
#     """Download the drivable road network from OpenStreetMap."""
#     print(f"🚦 Downloading road network for: {place_name}")
#     graph = ox.graph_from_place(place_name, network_type='drive')
#     edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
#     print(f"✅ Retrieved {len(edges)} road segments.")
#     return edges

# def fetch_shelter_pois(place_name: str) -> gpd.GeoDataFrame:
#     print(f"🏕️ Downloading potential shelter POIs for: {place_name}")
#     tags = {
#         'amenity': ['shelter', 'place_of_worship', 'school','hospital'],
#         'building': ['public', 'civic', 'school']
#     }
#     pois = ox.features_from_place(place_name, tags)

#     # الاحتفاظ بالأعمدة المفيدة فقط
#     useful_cols = ['geometry', 'name', 'amenity', 'building', 'leisure']
#     pois = pois[[col for col in useful_cols if col in pois.columns]].copy()

#     # تحويل كل كيان إلى نقطة (centroid)
#     # Project to metric (meters) then back
#     pois = pois.to_crs(epsg=3857)
#     pois['geometry'] = pois['geometry'].centroid
#     pois = pois.to_crs(epsg=4326)
#     pois = gpd.GeoDataFrame(pois, geometry='geometry', crs='EPSG:4326')

#     print(f"✅ Retrieved {len(pois)} POIs (converted to centroids).")
#     return pois


# def save_geodataframe(gdf: gpd.GeoDataFrame, output_path: str):
#     import os
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     # احتفظ فقط بأول عمود هندسي واسمه "geometry"
#     geom_cols = gdf.select_dtypes(include='geometry').columns.tolist()
#     first_geom = geom_cols[0]
    
#     gdf = gdf.drop(columns=[col for col in geom_cols if col != first_geom])
#     if first_geom != "geometry":
#         gdf = gdf.rename(columns={first_geom: "geometry"})

#     # إعادة بناء GeoDataFrame من نفس البيانات
#     gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

#     # الحفظ
#     gdf.to_file(f"{output_path}.shp")
#     gdf.to_file(f"{output_path}.geojson", driver="GeoJSON")
#     print(f"💾 Saved to: {output_path}.shp / .geojson")

import os
import osmnx as ox
import geopandas as gpd

def fetch_road_network(place_name: str) -> gpd.GeoDataFrame:
    """Download the drivable road network from OpenStreetMap."""
    print(f"🚦 Downloading road network for: {place_name}")
    graph = ox.graph_from_place(place_name, network_type='drive')
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    
    if edges.empty:
        print(f"⚠️ No road segments found for {place_name}.")
    else:
        print(f"✅ Retrieved {len(edges)} road segments.")
    
    return edges

def fetch_shelter_pois(place_name: str) -> gpd.GeoDataFrame:
    print(f"🏕️ Downloading potential shelter POIs for: {place_name}")
    tags = {
        'amenity': ['shelter', 'school', 'hospital'],  # Removed 'place_of_worship'
        'building': ['public', 'civic', 'school'],
        'leisure': ['park']  # Added 'park' for gardens
    }
    pois = ox.features_from_place(place_name, tags)

    # التأكد من وجود الأعمدة المطلوبة قبل استخدامها
    useful_cols = ['geometry', 'name', 'amenity', 'building', 'leisure']
    pois = pois[[col for col in useful_cols if col in pois.columns]].copy()

    # تحويل كل كيان إلى نقطة (centroid)
    if pois.shape[0] > 0:  # تحقق من وجود بيانات قبل المعالجة
        pois = pois.to_crs(epsg=3857)
        pois['geometry'] = pois['geometry'].centroid
        pois = pois.to_crs(epsg=4326)
        pois = gpd.GeoDataFrame(pois, geometry='geometry', crs='EPSG:4326')
    else:
        print("⚠️ No POIs found for this place.")

    print(f"✅ Retrieved {len(pois)} POIs (converted to centroids).")
    return pois


def save_geodataframe(gdf: gpd.GeoDataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # التأكد من وجود عمود هندسي قبل محاولة الوصول إليه
    geom_cols = gdf.select_dtypes(include='geometry').columns.tolist()
    if not geom_cols:
        print("⚠️ No geometry column found in the GeoDataFrame.")
        return

    first_geom = geom_cols[0]
    
    gdf = gdf.drop(columns=[col for col in geom_cols if col != first_geom])
    if first_geom != "geometry":
        gdf = gdf.rename(columns={first_geom: "geometry"})

    # إعادة بناء GeoDataFrame من نفس البيانات
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    # الحفظ
    gdf.to_file(f"{output_path}.shp")
    gdf.to_file(f"{output_path}.geojson", driver="GeoJSON")
    print(f"💾 Saved to: {output_path}.shp / .geojson")
