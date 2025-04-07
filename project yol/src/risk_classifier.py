# # src/risk_classifier.py

# import pandas as pd
# import geopandas as gpd
# import logging
# import os
# from shapely.geometry import Point
# from geopandas.tools import sjoin_nearest


# # Logger
# os.makedirs("logs", exist_ok=True)
# logging.basicConfig(filename="logs/risk_join.log", level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


# def classify_risk(input_csv, output_csv):
#     logging.info("🔎 Classifying seismic risk levels...")
#     df = pd.read_csv(input_csv)

#     if 'magnitude' not in df.columns:
#         raise ValueError("Input CSV must contain 'magnitude' column.")

#     bins = [0, 4.5, 5.5, 6.5, 10]
#     labels = ['Low', 'Medium', 'High', 'Critical']
#     df['risk_level'] = pd.cut(df['magnitude'], bins=bins, labels=labels)

#     os.makedirs(os.path.dirname(output_csv), exist_ok=True)
#     df.to_csv(output_csv, index=False)
#     logging.info(f"✅ Risk levels saved to {output_csv}")
#     return df



# def join_risk_with_shelters(
#     risk_fp="outputs/tables/risk_levels.csv",
#     shelters_fp="data/shapefiles/elazig_shelters.shp",
#     out_fp="outputs/tables/shelter_risk_joined.geojson"
# ):
#     logging.info("🔗 Starting risk classification join...")

#     gdf_shelters = gpd.read_file(shelters_fp)
#     gdf_risk = pd.read_csv(risk_fp)

#     gdf_risk = gpd.GeoDataFrame(
#         gdf_risk,
#         geometry=gpd.points_from_xy(gdf_risk.longitude, gdf_risk.latitude),
#         crs="EPSG:4326"
#     )

#     if gdf_shelters.crs != gdf_risk.crs:
#         gdf_shelters = gdf_shelters.to_crs(gdf_risk.crs)
#         logging.info("CRS aligned.")

#     joined = sjoin_nearest(gdf_shelters, gdf_risk, how="left")
#     logging.info(f"Joined {len(joined)} shelters with nearest risk levels.")

#     joined["type"] = joined.get("type", pd.Series(["shelter"] * len(joined))).fillna("shelter")
#     joined["image"] = joined.get("image", pd.Series(["https://cdn-icons-png.flaticon.com/512/684/684908.png"] * len(joined))).fillna("https://cdn-icons-png.flaticon.com/512/684/684908.png")
#     joined["description"] = joined.get("description", pd.Series(["تم تصنيفه كخيار إخلاء طارئ في حالات الزلازل."] * len(joined))).fillna("تم تصنيفه كخيار إخلاء طارئ في حالات الزلازل.")

#     os.makedirs(os.path.dirname(out_fp), exist_ok=True)
#     joined.to_file(out_fp, driver="GeoJSON")
#     logging.info(f"✅ Output saved: {out_fp}")
import pandas as pd
import geopandas as gpd
import logging
import os
from shapely.geometry import Point
from geopandas.tools import sjoin_nearest


# Logger
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/risk_join.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def classify_risk(input_csv, output_csv):
    logging.info("🔎 Classifying seismic risk levels...")

    # قراءة بيانات CSV
    df = pd.read_csv(input_csv)

    # التأكد من وجود العمود 'magnitude'
    if 'magnitude' not in df.columns:
        raise ValueError("Input CSV must contain 'magnitude' column.")
    
    # إضافة فئات المخاطر بناءً على شدة الزلزال
    bins = [0, 4.5, 5.5, 6.5, 10]
    labels = ['Low', 'Medium', 'High', 'Critical']
    df['risk_level'] = pd.cut(df['magnitude'], bins=bins, labels=labels)

    # حفظ النتيجة في ملف CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logging.info(f"✅ Risk levels saved to {output_csv}")
    return df


def join_risk_with_shelters(
    risk_fp="outputs/tables/risk_levels.csv",
    shelters_fp="data/shapefiles/elazig_shelters.shp",
    out_fp="outputs/tables/shelter_risk_joined.geojson"
):
    logging.info("🔗 Starting risk classification join...")

    # قراءة ملفات البيانات
    gdf_shelters = gpd.read_file(shelters_fp)
    gdf_risk = pd.read_csv(risk_fp)

    # التأكد من وجود الأعمدة 'longitude' و 'latitude' في بيانات المخاطر
    if 'longitude' not in gdf_risk.columns or 'latitude' not in gdf_risk.columns:
        raise ValueError("Risk CSV must contain 'longitude' and 'latitude' columns.")

    # تحويل بيانات المخاطر إلى GeoDataFrame باستخدام الأعمدة الجغرافية
    gdf_risk = gpd.GeoDataFrame(
        gdf_risk,
        geometry=gpd.points_from_xy(gdf_risk.longitude, gdf_risk.latitude),
        crs="EPSG:4326"
    )

    # التأكد من تطابق النظام الإحداثي (CRS) بين بيانات الملاجئ والمخاطر
    if gdf_shelters.crs != gdf_risk.crs:
        gdf_shelters = gdf_shelters.to_crs(gdf_risk.crs)
        logging.info("CRS aligned.")

    # دمج بيانات المخاطر مع بيانات الملاجئ باستخدام أقرب نقطة
    joined = sjoin_nearest(gdf_shelters, gdf_risk, how="left")
    logging.info(f"Joined {len(joined)} shelters with nearest risk levels.")

    # إضافة خصائص افتراضية إذا كانت مفقودة
    joined["type"] = joined.get("type", pd.Series(["shelter"] * len(joined))).fillna("shelter")
    joined["image"] = joined.get("image", pd.Series(["https://cdn-icons-png.flaticon.com/512/684/684908.png"] * len(joined))).fillna("https://cdn-icons-png.flaticon.com/512/684/684908.png")
    joined["description"] = joined.get("description", pd.Series(["تم تصنيفه كخيار إخلاء طارئ في حالات الزلازل."] * len(joined))).fillna("تم تصنيفه كخيار إخلاء طارئ في حالات الزلازل.")

    # حفظ النتائج في ملف GeoJSON
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    joined.to_file(out_fp, driver="GeoJSON")
    logging.info(f"✅ Output saved: {out_fp}")
