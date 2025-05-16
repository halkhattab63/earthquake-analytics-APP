import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

def fetch_earthquakes_usgs(start_year=2020, end_year=2024, min_magnitude=1):
    all_rows = []

    for year in range(start_year, end_year + 1):
        print(f" Fetching year {year}...")
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            "format": "geojson",
            "starttime": f"{year}-01-01",
            "endtime": f"{year}-12-31",
            "minmagnitude": min_magnitude,
            "limit": 2000
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            for feature in data["features"]:
                props = feature["properties"]
                coords = feature["geometry"]["coordinates"]
                all_rows.append({
                    "time": props.get("time"),
                    "place": props.get("place"),
                    "magnitude": props.get("mag"),
                    "latitude": coords[1],
                    "longitude": coords[0],
                    "depth_km": coords[2]
                })

        except Exception as e:
            print(f" Error fetching {year}: {e}")

    df = pd.DataFrame(all_rows)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    print(f" Total earthquakes fetched: {len(df)}")
    return df

#Türkiye'nin coğrafi sınırları
def filter_inside_turkey(df, geojson_path="tr.json"):
    print(" Filtering earthquakes inside Turkish borders...")
    try:
        turkey = gpd.read_file(geojson_path)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=turkey.crs)
        filtered = gdf[gdf.within(turkey.unary_union)].copy()
        filtered.drop(columns=['geometry'], inplace=True)
        print(f" Inside Turkey: {len(filtered)} / {len(df)} earthquakes")
        return filtered
    except Exception as e:
        print(f" GeoJSON load/filtering failed: {e}")
        return df

if __name__ == "__main__":
    raw_df = fetch_earthquakes_usgs()
    df_turkey = filter_inside_turkey(raw_df, geojson_path="tr.json")

    os.makedirs("data", exist_ok=True)
    
    # Filtrelenmiş Veri Çerçevesini CSV'ye kaydedin
    df_turkey.to_csv("earthquake_risk_predictor/data/dataset/usgs_earthquakes_turkey2020.csv", index=False)
    print(" Saved to data/usgs_earthquakes_turkey.csv")
