# map_functions.py
import folium
from folium.plugins import MarkerCluster, HeatMap, Fullscreen, TimestampedGeoJson
from branca.colormap import linear
import streamlit as st  # Make sure streamlit is imported for error handling
from APP.color_utils import get_color_by_magnitude
def create_interactive_map(data):
    """Depremler için etkileşimli harita oluştur"""
    if data.empty:
        return None
    
    m = folium.Map(
        location=[data["latitude"].mean(), data["longitude"].mean()],
        zoom_start=6,
        tiles="cartodbpositron",
        control_scale=True
    )
    
    colormap = linear.YlOrRd_09.scale(data["magnitude"].min(), data["magnitude"].max())
    colormap.caption = "Deprem Şiddeti (Ritcher Ölçeği)"
    colormap.add_to(m)
    
    marker_cluster = MarkerCluster(
        name="Nokta Kümeleme",
        overlay=True,
        control=True
    ).add_to(m)
    
    for _, row in data.iterrows():
        popup_content = f"""
        <div style="width: 250px;">
            <h4 style="color: #e74c3c; margin-bottom: 5px;">Deprem Bilgileri</h4>
            <p><b>Konum:</b> {row['place']}</p>
            <p><b>İl:</b> {row['province']}</p>
            <p><b>Tarih:</b> {row['time'].strftime('%Y-%m-%d %H:%M')}</p>
            <p><b>Şiddet:</b> {row['magnitude']}</p>
            <p><b>Sınıf:</b> {row['severity']}</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5 + row["magnitude"],
            color=colormap(row["magnitude"]),
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=f"Şiddet: {row['magnitude']} - {row['place']}"
        ).add_to(marker_cluster)
    
    # Ek kontrol öğeleri ekleme
    Fullscreen(position="topright").add_to(m)
    folium.LayerControl().add_to(m)
    
    return m
def create_heatmap(data):
    """Depremler için ısı haritası oluştur"""
    if data.empty:
        return None
    
    heat_map = folium.Map(
        location=[data["latitude"].mean(), data["longitude"].mean()],
        zoom_start=6,
        tiles="CartoDB.DarkMatter"
    )
    
    heat_data = data[["latitude", "longitude"]].values.tolist()
    HeatMap(
        heat_data,
        radius=15,
        blur=10,
        gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'orange', 1.0: 'red'}
    ).add_to(heat_map)
    
    Fullscreen(position="topright").add_to(heat_map)
    folium.LayerControl().add_to(heat_map)
    
    return heat_map
def create_time_animation(data):
    """Depremler için zaman animasyonu haritası oluştur"""
    if data.empty:
        return None
    
    try:
        # Tarihi ISO formatına dönüştür
        data['time_iso'] = data['time'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        features = []
        for _, row in data.iterrows():
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["longitude"], row["latitude"]],
                },
                "properties": {
                    "time": row["time_iso"],
                    "style": {"color": get_color_by_magnitude(row['magnitude'])},
                    "icon": "circle",
                    "iconstyle": {
                        "fillColor": get_color_by_magnitude(row['magnitude']),
                        "fillOpacity": 0.7,
                        "stroke": True,
                        "radius": 5 + row["magnitude"]
                    },
                    "popup": f"<b>📍 {row['place']}</b><br>📅 {row['time'].date()}<br>🌡️ Şiddet: {row['magnitude']}"
                }
            })

        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }

        m = folium.Map(
            location=[data["latitude"].mean(), data["longitude"].mean()],
            zoom_start=6,
            tiles="cartodbpositron"
        )

        TimestampedGeoJson(
            geojson_data,
            period="P1M",  # Her kare arasındaki zaman aralığı (aylık)
            add_last_point=True,
            auto_play=True,
            loop=False,
            max_speed=1,
            loop_button=True,
            date_options="YYYY/MM/DD",
            time_slider_drag_update=True
        ).add_to(m)

        return m
    except Exception as e:
        st.error(f"Zaman animasyonu oluşturulurken hata oluştu: {str(e)}")
        return None
