# src/dashboard.py

import os
import logging
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import geopandas as gpd
import plotly.express as px
import folium
from folium.plugins import MarkerCluster

# Initialize logging and create necessary directories
LOG_DIR = "earthquake analytics APP/project yol/logs"
MAP_DIR = "earthquake analytics APP/project yol/outputs/maps"
TABLES_DIR = "earthquake analytics APP/project yol/outputs/tables"
DATA_DIR = "earthquake analytics APP/project yol/data/cleaned"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MAP_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "dashboard.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def generate_risk_map_html(
    geojson_path=os.path.join(TABLES_DIR, "shelter_risk_joined.geojson"),
    out_path=os.path.join(MAP_DIR, "shelter_risk_map.html")
):
    logging.info("Generating colored shelter map with icons and details...")

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

        icon = icon_mapping.get(place_type, "info-sign")
        color = "red" if risk == "High" else "orange" if risk == "Medium" else "green"

        popup_html = f"""
        <div style='width:220px'>
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
    logging.info(f"Map with shelters saved to {out_path}")
    return out_path

def build_dashboard():
    logging.info("Building modern dashboard...")

    quake_file = os.path.join(DATA_DIR, "earthquakes_cleaned.csv")
    shelter_file = os.path.join(TABLES_DIR, "shelter_risk_joined.geojson")

    quakes = pd.read_csv(quake_file)
    shelters = gpd.read_file(shelter_file)

    total_shelters = len(shelters)
    risk_counts = shelters["risk_level"].value_counts().to_dict()

    risk_map_path = generate_risk_map_html()

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    app.title = "Earthquake Risk Dashboard"

    app.layout = dbc.Container([
        dbc.NavbarSimple(
            brand="Earthquake & Shelter Dashboard",
            color="primary", dark=True, className="mb-4"
        ),

        dbc.Row([
            dbc.Col(dbc.Card([
                html.H4("Total Shelters"),
                html.H2(f"{total_shelters}")
            ], body=True, color="secondary"), width=3),

            *[
                dbc.Col(dbc.Card([
                    html.H4(f"{level} Risk"),
                    html.H2(f"{count}")
                ], body=True,
                   color="danger" if level == "High" else "warning" if level == "Medium" else "success"), width=3)
                for level, count in risk_counts.items()
            ]
        ], className="mb-4"),

        dcc.Tabs([
            dcc.Tab(label="Shelter Risk Map", children=[
                html.Iframe(
                    srcDoc=open(risk_map_path, encoding="utf-8").read(),
                    width="100%", height="600",
                    style={"border": "1px solid #aaa", "borderRadius": "8px"}
                )
            ]),
            dcc.Tab(label="Earthquake Statistics", children=[
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

    logging.info("Dashboard generated successfully.")
    return app