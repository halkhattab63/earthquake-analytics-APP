import geopandas as gpd
from shapely.geometry import Point
from src.evacuation import build_road_graph, calculate_evacuation_path

def test_evacuation_path_generation():
    roads = gpd.read_file("data/shapefiles/elazig_roads.shp")
    start = (38.675, 39.221)
    end = (38.685, 39.235)
    gdf_path = calculate_evacuation_path(roads, start, end)
    assert not gdf_path.empty, "Evacuation path GeoDataFrame is empty."
