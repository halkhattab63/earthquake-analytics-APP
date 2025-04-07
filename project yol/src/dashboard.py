# import os
# import dash
# from dash import dcc, html
# import dash_bootstrap_components as dbc
# import pandas as pd
# import geopandas as gpd
# import plotly.express as px
# import folium
# from folium.plugins import MarkerCluster
# import logging

# # إعداد مجلدات الحفظ والتسجيل
# os.makedirs("logs", exist_ok=True)
# os.makedirs("outputs/maps", exist_ok=True)

# logging.basicConfig(filename="logs/dashboard.log", level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# def generate_risk_map_html(
#     geojson_path="outputs/tables/shelter_risk_joined.geojson",
#     out_path="outputs/maps/shelter_risk_map.html"
# ):
#     logging.info("🎨 Generating colored shelter map with icons and details...")

#     shelters = gpd.read_file(geojson_path)
#     fmap = folium.Map(location=[38.67, 39.22], zoom_start=13, control_scale=True)
#     cluster = MarkerCluster().add_to(fmap)

#     icon_mapping = {
#         "school": "graduation-cap",
#         "hospital": "plus-square",
#         "tent": "campground",
#         "park": "tree",
#         "shelter": "home"
#     }

#     for _, row in shelters.iterrows():
#         lat, lon = row.geometry.y, row.geometry.x
#         risk = row.get("risk_level", "Unknown")
#         place_type = row.get("type", "shelter").lower()
#         name = row.get("name", "Unnamed")
#         desc = row.get("description", "No details")
#         image = row.get("image", "https://cdn-icons-png.flaticon.com/512/684/684908.png")

#         icon = icon_mapping.get(place_type, "info-sign")
#         color = "red" if risk == "High" else "orange" if risk == "Medium" else "green"

#         popup_html = f"""
#         <div style="width:220px">
#             <h4>{name}</h4>
#             <img src="{image}" width="100%" style="border-radius:4px"><br>
#             <b>Type:</b> {place_type.capitalize()}<br>
#             <b>Risk Level:</b> {risk}<br>
#             <small>{desc}</small>
#         </div>
#         """

#         folium.Marker(
#             location=[lat, lon],
#             popup=folium.Popup(popup_html, max_width=250),
#             icon=folium.Icon(color=color, icon=icon, prefix="fa")
#         ).add_to(cluster)

#     folium.LayerControl(collapsed=False).add_to(fmap)
#     fmap.save(out_path)
#     logging.info(f"✅ Map with shelters saved to {out_path}")
#     return out_path


# def build_dashboard():
#     logging.info("🧩 Building modern dashboard...")

#     # Load data
#     quakes = pd.read_csv("data/cleaned/earthquakes_cleaned.csv")
#     shelters = gpd.read_file("outputs/tables/shelter_risk_joined.geojson")

#     total = len(shelters)
#     risk_counts = shelters["risk_level"].value_counts().to_dict()

#     # Generate map
#     risk_map_path = generate_risk_map_html()

#     # Init app
#     app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
#     app.title = "Earthquake Risk Dashboard"

#     app.layout = dbc.Container([
#         dbc.NavbarSimple(
#             brand="🌍 Earthquake & Shelter Dashboard",
#             color="primary", dark=True, className="mb-4"
#         ),

#         dbc.Row([
#             dbc.Col(dbc.Card([
#                 html.H4("📍 Total Shelters"),
#                 html.H2(f"{total}")
#             ], body=True, color="secondary"), width=3),

#             *[
#                 dbc.Col(dbc.Card([
#                     html.H4(f"{level} Risk"),
#                     html.H2(f"{count}")
#                 ], body=True, color="danger" if level == "High" else "warning" if level == "Medium" else "success"), width=3)
#                 for level, count in risk_counts.items()
#             ]
#         ], className="mb-4"),

#         dcc.Tabs([
#             dcc.Tab(label="🗺️ Shelter Risk Map", children=[
#                 html.Iframe(
#                     srcDoc=open(risk_map_path, encoding="utf-8").read(),
#                     width="100%", height="600",
#                     style={"border": "1px solid #aaa", "borderRadius": "8px"}
#                 )
#             ]),
#             dcc.Tab(label="📈 Earthquake Statistics", children=[
#                 dcc.Graph(
#                     figure=px.histogram(quakes, x="magnitude", nbins=20,
#                                         title="Earthquake Magnitude Distribution",
#                                         template="plotly_dark")
#                 )
#             ])
#         ])
#     ], fluid=True)

#     logging.info("✅ Dashboard generated successfully.")
#     return app
# src/generate_maps.py

# src/dashboard.py

import os
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import geopandas as gpd
import plotly.express as px
import logging
import folium
from folium.plugins import MarkerCluster

# إعداد مجلدات الحفظ والتسجيل
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs/maps", exist_ok=True)

logging.basicConfig(filename="logs/dashboard.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def generate_risk_map_html(
    geojson_path="outputs/tables/shelter_risk_joined.geojson",
    out_path="outputs/maps/shelter_risk_map.html"
):
    logging.info("🎨 Generating colored shelter map with icons and details...")

    shelters = gpd.read_file(geojson_path)
    fmap = folium.Map(location=[38.67, 39.22], zoom_start=13, control_scale=True)
    cluster = MarkerCluster().add_to(fmap)

    icon_mapping = {
        "school": "graduation-cap",
        "hospital": "plus-square",
        "tent": "campground",
        "park": "tree",
        "shelter": "home"
    }

    for _, row in shelters.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        risk = row.get("risk_level", "Unknown")
        place_type = str(row.get("type", "shelter")).lower()
        name = row.get("name", "Unnamed")
        desc = row.get("description", "No details")
        # image = row.get("image", "https://cdn-icons-png.flaticon.com/512/684/684908.png")

        icon = icon_mapping.get(place_type, "info-sign")
        color = "red" if risk == "High" else "orange" if risk == "Medium" else "green"
        # <img src="{image}" width="100%" style="border-radius:4px"><br>
        popup_html = f"""
        <div style="width:220px">
            <h4>{name}</h4>



            <b>Type:</b> {place_type.capitalize()}<br>
            <b>Risk Level:</b> {risk}<br>
            <small>{desc}</small>
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color=color, icon=icon, prefix="fa")
        ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(out_path)
    logging.info(f"✅ Map with shelters saved to {out_path}")
    return out_path


def build_dashboard():
    logging.info("🧩 Building modern dashboard...")

    quakes = pd.read_csv("data/cleaned/earthquakes_cleaned.csv")
    shelters = gpd.read_file("outputs/tables/shelter_risk_joined.geojson")

    total = len(shelters)
    risk_counts = shelters["risk_level"].value_counts().to_dict()

    # توليد خريطة تفاعلية
    risk_map_path = generate_risk_map_html()

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    app.title = "Earthquake Risk Dashboard"

    app.layout = dbc.Container([
        dbc.NavbarSimple(
            brand="🌍 Earthquake & Shelter Dashboard",
            color="primary", dark=True, className="mb-4"
        ),

        dbc.Row([
            dbc.Col(dbc.Card([
                html.H4("📍 Total Shelters"),
                html.H2(f"{total}")
            ], body=True, color="secondary"), width=3),

            *[
                dbc.Col(dbc.Card([
                    html.H4(f"{level} Risk"),
                    html.H2(f"{count}")
                ], body=True, color="danger" if level == "High" else "warning" if level == "Medium" else "success"), width=3)
                for level, count in risk_counts.items()
            ]
        ], className="mb-4"),

        dcc.Tabs([
            dcc.Tab(label="🗺️ Shelter Risk Map", children=[
                html.Iframe(
                    srcDoc=open(risk_map_path, encoding="utf-8").read(),
                    width="100%", height="600",
                    style={"border": "1px solid #aaa", "borderRadius": "8px"}
                )
            ]),
            dcc.Tab(label="📈 Earthquake Statistics", children=[
                dcc.Graph(
                    figure=px.histogram(
                        quakes, x="magnitude", nbins=20,
                        title="Earthquake Magnitude Distribution",
                        template="plotly_dark"
                    )
                )
            ])
        ])
    ], fluid=True)

    logging.info("✅ Dashboard generated successfully.")
    return app
