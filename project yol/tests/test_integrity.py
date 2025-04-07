import geopandas as gpd
import pandas as pd

def test_earthquake_count():
    df = pd.read_csv("data/cleaned/earthquakes_cleaned.csv")
    assert len(df) > 1000, "🚨 عدد الزلازل قليل جدًا!"

def test_shelter_count():
    shelters = gpd.read_file("data/shapefiles/elazig_shelters.shp")
    assert len(shelters) > 100, "🚨 عدد الملاجئ غير كافٍ!"

def test_evacuation_path_length():
    path = gpd.read_file("outputs/shapefiles/evacuation_path.shp")
    assert len(path.geometry[0].coords) > 50, "🚨 المسار قصير جدًا!"
