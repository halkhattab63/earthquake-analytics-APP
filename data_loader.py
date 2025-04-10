# data_loader.py
import pandas as pd
from datetime import datetime
import streamlit as st
import sys
sys.stdout.reconfigure(encoding='utf-8')

@st.cache_data
def load_data():
    """Depremleri yükleyin ve temizleyin"""
    try:
        df = pd.read_csv("cleaned_earthquake_data_turkey.csv")
        
        # Daha kapsamlı temizlik
        df = df.dropna(subset=["latitude", "longitude", "magnitude", "time", "place", "province"])
        
        # Tarih dönüşümü
        df["time"] = pd.to_datetime(df["time"])
        
        # Veri türlerinin düzeltilmesi
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
        
        # Geçersiz değerlerin kaldırılması
        df = df[
            (df["latitude"].between(-90, 90)) & 
            (df["longitude"].between(-180, 180)) & 
            (df["magnitude"] > 0)
        ]
        
        # Tehlike sınıfı eklenmesi
        df['severity'] = pd.cut(df['magnitude'],
                               bins=[0, 3, 4, 5, 6, 10],
                               labels=['Hafif', 'Orta', 'Güçlü', 'Şiddetli', 'Felaket'],
                               right=False)
        return df
    except Exception as e:
        st.error(f"Veri yükleme hatası oluştu: {str(e)}")
        return pd.DataFrame()
