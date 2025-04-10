# ui.py
import streamlit as st
import pandas as pd
from filters import apply_filters

def show_sidebar(df):
    """Yan menüyü göster ve filtrelerle etkileşimde bulun"""
    st.sidebar.title("🧭 Kontrol Paneli")
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Flag_of_Turkey.svg/1200px-Flag_of_Turkey.svg.png", width=100)

    selected_tab = st.sidebar.radio(
        "Analiz bölümünü seçin:",
        options=[
            "🗺️ Etkileşimli Harita",
            "🌡️ Isı Haritası",
            "📊 Analiz Paneli",
            "📈 Zaman İstatistikleri",
            "🎥 Zaman Sırası",
            
            "🧠 Yapay Zeka Tahminleri", 
            "🗺️ Sığınak ve Risk Haritası",
            "📤 Veri Dışa Aktarımı",
            "ℹ️ Proje Hakkında"
        ]
    )

    st.sidebar.header("🔍 Verileri Filtrele")
    province_options = ["Tümü"] + sorted(df["province"].dropna().unique())
    selected_province = st.sidebar.selectbox("🏙️ İl Seçin", province_options)

    mag_min, mag_max = float(df["magnitude"].min()), float(df["magnitude"].max())
    mag_range = st.sidebar.slider(
        "📊 Şiddet Aralığı (Ritcher Ölçeği)",
        min_value=mag_min,
        max_value=mag_max,
        value=(mag_min, mag_max),
        step=0.1
    )

    date_min, date_max = df["time"].min().to_pydatetime(), df["time"].max().to_pydatetime()
    date_range = st.sidebar.slider(
        "📅 Zaman Aralığı",
        min_value=date_min,
        max_value=date_max,
        value=(date_min, date_max),
        format="YYYY-MM-DD"
    )

    severity_options = ["Tümü"] + list(df["severity"].cat.categories)
    selected_severity = st.sidebar.selectbox("⚠️ Tehlike Sınıfı", severity_options)

    filtered_df = apply_filters(df, mag_range, date_range, selected_province, selected_severity)

    return selected_tab, filtered_df
