# filters.py
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')
def apply_filters(df, mag_range, date_range, selected_province, selected_severity):
    """
    Verileri belirtilen filtrelere göre filtreleme:
    - Şiddet aralığı (Magnitude)
    - Tarih aralığı (Date)
    - İl (Province)
    - Tehlike sınıfı (Severity)
    
    :param df: Orijinal veri (DataFrame)
    :param mag_range: Şiddet aralığı (tuple)
    :param date_range: Tarih aralığı (tuple)
    :param selected_province: Seçilen il (str)
    :param selected_severity: Seçilen tehlike sınıfı (str)
    :return: Filtrelenmiş veri (DataFrame)
    """
    # Şiddet aralığına göre veriyi filtreleme
    filtered_df = df[
        (df["magnitude"] >= mag_range[0]) & 
        (df["magnitude"] <= mag_range[1]) & 
        (df["time"] >= pd.to_datetime(date_range[0])) & 
        (df["time"] <= pd.to_datetime(date_range[1]))
    ]
    
    # İl'e göre veriyi filtreleme
    if selected_province != "Tümü":
        filtered_df = filtered_df[filtered_df["province"] == selected_province]

    # Tehlike sınıfına göre veriyi filtreleme
    if selected_severity != "Tümü":
        filtered_df = filtered_df[filtered_df["severity"] == selected_severity]
    
    return filtered_df
