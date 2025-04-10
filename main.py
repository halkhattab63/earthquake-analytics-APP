# main.py
import streamlit as st
import pandas as pd
from app_config import configure_app
from data_loader import load_data
from pdf_report import generate_pdf
from map_functions import create_interactive_map, create_heatmap
from ui import show_sidebar
from filter_s import apply_filters
from tabs import display_tab
# CSS ile görünümü özelleştirme
def load_css():
    # Dosyayı UTF-8 kodlamasıyla açtığınızdan emin olun
    with open("styles.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Uygulama ayarlarını yapılandırma
configure_app()

# CSS dosyasını yükle
load_css()


# Verileri yükle
df = load_data()

# Filtreler için yan menüyü göster
selected_tab, filtered_df = show_sidebar(df)

# Seçilen sekmeyi filtrelere göre göster
display_tab(selected_tab, filtered_df)
