# tabs.py
from io import BytesIO
import os
import folium
import pandas as pd
import streamlit as st
from folium.plugins import MarkerCluster, HeatMap, Fullscreen
from map_functions import create_interactive_map, create_heatmap
import plotly.express as px
from streamlit_folium import st_folium
import plotly.graph_objects as go
from pdf_report import generate_pdf
def display_tab(selected_tab, filtered_df):
    """Seçilen sekmeleri göster"""
    
    # Etkileşimli harita sekmesi
    if selected_tab == "🗺️ Etkileşimli Harita":
        st.title("📍 Türkiye'deki Depremler İçin Etkileşimli Harita")
        if not filtered_df.empty:
            with st.spinner("Harita yükleniyor..."):
                map_obj = create_interactive_map(filtered_df)
                st_folium(map_obj, width=1200, height=700)
            st.subheader("Veri Görüntüleme")
            st.dataframe(filtered_df.head(100))
        else:
            st.warning("⚠️ Seçilen filtrelerle eşleşen veri bulunmamaktadır.")
    
    # Isı haritası sekmesi
    elif selected_tab == "🌡️ Isı Haritası":
        st.header("🌡️ Depremlerin Dağılımı İçin Isı Haritası")
        if not filtered_df.empty:
            heat_data = filtered_df[["latitude", "longitude"]].dropna().values.tolist()
            heat_map = folium.Map(location=[filtered_df["latitude"].mean(), filtered_df["longitude"].mean()], zoom_start=6, tiles="CartoDB.DarkMatter")
            HeatMap(heat_data, radius=15, blur=10).add_to(heat_map)
            Fullscreen().add_to(heat_map)
            st_folium(heat_map, width=1200, height=650)
        else:
            st.warning("⚠️ Harita çizmek için yeterli veri yok.")

    # Analiz paneli sekmesi
    elif selected_tab == "📊 Analiz Paneli":
        st.title("📊 Deprem Analiz Paneli")
        if not filtered_df.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Toplam Deprem Sayısı", len(filtered_df))
            with col2:
                st.metric("En Şiddetli Deprem", f"{filtered_df['magnitude'].max():.1f}")
            with col3:
                st.metric("Son Deprem", filtered_df['time'].max().strftime('%Y-%m-%d'))
            
            # İl bazında analiz
            st.subheader("📈 İl Bazında Analiz")
            province_stats = filtered_df.groupby("province").agg(
                deprem_sayisi=("magnitude", "count"),
                ortalama_siddet=("magnitude", "mean"),
                maksimum_siddet=("magnitude", "max")
            ).sort_values("deprem_sayisi", ascending=False)
            fig1 = px.bar(
                province_stats,
                x=province_stats.index,
                y="deprem_sayisi",
                title="İllere Göre Deprem Sayısı",
                color="ortalama_siddet",
                color_continuous_scale="OrRd"
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Sınıf bazında analiz
            st.subheader("📊 Sınıf Bazında Analiz")
            severity_stats = filtered_df["severity"].value_counts().sort_index()
            fig2 = px.pie(
                severity_stats,
                names=severity_stats.index,
                values=severity_stats.values,
                title="Tehlike Seviyesine Göre Depremlerin Dağılımı"
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("🗃️ Detaylı Veriler")
            st.dataframe(filtered_df)

        else:
            st.warning("⚠️ Gösterilecek veri bulunmamaktadır.")
    # Zaman İstatistikleri sekmesi
    elif selected_tab == "📈 Zaman İstatistikleri":
        st.title("📈 Deprem Aktivitesinin Zaman Analizi")
        if not filtered_df.empty:
            time_df = filtered_df.copy()
            time_df["year"] = time_df["time"].dt.year
            time_df["month"] = time_df["time"].dt.month
            time_df["year_month"] = time_df["time"].dt.to_period("M").astype(str)

            # Yıllık dağılım
            st.subheader("Yıllık Dağılım")
            yearly = time_df["year"].value_counts().sort_index()
            fig1 = px.line(
                yearly,
                x=yearly.index,
                y=yearly.values,
                title="Yıllık Deprem Sayısı",
                labels={"x": "Yıl", "y": "Deprem Sayısı"}
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Aylık dağılım
            st.subheader("Aylık Dağılım")
            monthly = time_df.groupby(["year", "month"]).size().reset_index(name="count")
            fig2 = px.line(
                monthly,
                x="month",
                y="count",
                color="year",
                title="Yıllara Göre Aylık Deprem Sayısı",
                labels={"month": "Ay", "count": "Deprem Sayısı"}
            )
            st.plotly_chart(fig2, use_container_width=True)

        else:
            st.warning("⚠️ Zaman analizi için yeterli veri yok.")

    # Zaman sırası sekmesi
    elif selected_tab == "🎥 Zaman Sırası":
        st.title("🎥 Deprem Aktivitesinin Zaman Sırası")
        if not filtered_df.empty:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
                        padding: 20px;
                        border-radius: 12px;
                        margin-bottom: 25px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        border-left: 6px solid #e74c3c;
                        color: white;">
                <h4 style="color: #f1c40f; margin-top: 0; margin-bottom: 15px;">🎥 Hareketli Harita Kullanım Kılavuzu</h4>
                <p>Hareketli harita, deprem aktivitesinin zaman içindeki gelişimini gösterir. Şunları yapabilirsiniz:</p>
                <ul>
                    <li>Hareketi başlatmak için oynat düğmesini kullanın</li>
                    <li>Hareketi durdurmak için durdur düğmesini kullanın</li>
                    <li>Zaman çubuğunu sürükleyerek farklı dönemleri gezin</li>
                    <li>Haritayı büyütüp küçülterek detayları keşfedin</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Hareketli harita hazırlanıyor..."):
                try:
                    time_df = filtered_df.copy()
                    time_df['year_month'] = time_df['time'].dt.to_period('M').astype(str)

                    # scatter_mapbox kullanarak hareketli harita oluştur
                    fig = px.scatter_mapbox(
                        time_df,
                        lat='latitude',
                        lon='longitude',
                        color='magnitude',
                        size='magnitude',
                        hover_name='place',
                        hover_data={'province': True, 'time': True, 'magnitude': ':.1f'},
                        animation_frame=time_df['time'].dt.strftime('%Y-%m'),
                        center=dict(lat=time_df['latitude'].mean(), lon=time_df['longitude'].mean()),
                        zoom=5,
                        height=600,
                        mapbox_style="carto-positron",  # Profesyonel harita
                        title='Türkiye’deki Depremlerin Zaman Dağılımı',
                        color_continuous_scale='Viridis',
                        size_max=10,
                        opacity=0.6
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Hareketli harita oluşturulurken hata oluştu: {str(e)}")

        else:
            st.warning("⚠️ Zaman sırasını göstermek için veri yok")

    # Veri dışa aktarımı sekmesi
    # Veri dışa aktarımı sekmesi
    elif selected_tab == "📤 Veri Dışa Aktarımı":
        st.title("📤 Veri Dışa Aktarımı")

        if not filtered_df.empty:
            st.subheader("📁 Filtrelenmiş Veriyi Dışa Aktar")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**📄 CSV Olarak İndir**")
                csv = filtered_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSV İndir",
                    data=csv,
                    file_name="Depremler_Turkiye.csv",
                    mime="text/csv"
                )

            with col2:
                st.markdown("**📊 Excel Olarak İndir**")
                from io import BytesIO
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    filtered_df.to_excel(writer, index=False, sheet_name="Depremler")
                st.download_button(
                    label="Excel İndir",
                    data=excel_buffer.getvalue(),
                    file_name="Depremler_Turkiye.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            with col3:
                st.markdown("**📑 PDF Olarak İndir**")
                pdf_buffer = generate_pdf(filtered_df)
                st.download_button(
                    label="PDF İndir",
                    data=pdf_buffer,
                    file_name="Deprem_Raporu.pdf",
                    mime="application/pdf"
                )

            st.subheader("Veri Önizlemesi")
            st.dataframe(filtered_df.head(50))
        else:
            st.warning("⚠️ Dışa aktarılacak veri bulunmamaktadır.")

        # 🔽 Ek Raporlar: Model çıktıları
        st.divider()
        st.subheader("🧠 Yapay Zeka Raporları ve Tahminler")

        col_pdf, col_html, col_ai = st.columns(3)

        with col_pdf:
            if os.path.exists("data/earthquake_report.pdf"):
                with open("data/earthquake_report.pdf", "rb") as f:
                    st.download_button(
                        label="📄 Model PDF Raporu",
                        data=f,
                        file_name="earthquake_report.pdf",
                        mime="application/pdf"
                    )

        with col_html:
            if os.path.exists("data/report.html"):
                with open("data/report.html", "rb") as f:
                    st.download_button(
                        label="🌐 Model HTML Raporu",
                        data=f,
                        file_name="earthquake_report.html",
                        mime="text/html"
                    )

        with col_ai:
            if os.path.exists("data/predictions.csv"):
                with open("data/predictions.csv", "rb") as f:
                    st.download_button(
                        label="📊 Tahminler CSV",
                        data=f,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

    elif selected_tab == "🧠 Yapay Zeka Tahminleri":
        st.title("🧠 AI ile Deprem Tahmini ve Analizi")
        try:
            base_path = "earthquake_risk_predictor/data/"
            pred_path = os.path.join(base_path, "predictions.csv")

            if os.path.exists(pred_path):
                df_pred = pd.read_csv(pred_path)

                st.subheader("📋 Tahmin Verileri ")
                st.dataframe(df_pred.head(50))

                st.subheader("📉 Tahmin Hataları")
                err_img = os.path.join(base_path, "regression_errors.png")
                if os.path.exists(err_img):
                    st.image(err_img, use_container_width=True)
                else:
                    st.warning("📉 regression_errors.png bulunamadı.")

                st.subheader("📊 Performans Ölçümleri")
                col1, col2 = st.columns(2)

                with col1:
                    reg_metrics_path = os.path.join(base_path, "regression_metrics.csv")
                    if os.path.exists(reg_metrics_path):
                        st.write("📈 Regression Metrics")
                        st.dataframe(pd.read_csv(reg_metrics_path))

                with col2:
                    xgb_metrics_path = os.path.join(base_path, "xgboost_metrics.csv")
                    if os.path.exists(xgb_metrics_path):
                        st.write("📈 XGBoost Metrics")
                        st.dataframe(pd.read_csv(xgb_metrics_path))

                st.subheader("🧩 Ek Görseller")
                col1, col2 = st.columns(2)

                with col1:
                    roc_path = os.path.join(base_path, "roc_curve.png")
                    if os.path.exists(roc_path):
                        st.image(roc_path, caption="ROC Curve", use_container_width=True)

                with col2:
                    cm_path = os.path.join(base_path, "confusion_matrix.png")
                    if os.path.exists(cm_path):
                        st.image(cm_path, caption="Confusion Matrix", use_container_width=True)

                st.subheader("📊 XGBoost Tahmin Görselleri")
                col3, col4 = st.columns(2)
                xgb_mag_path = os.path.join(base_path, "xgboost_results_magnitude.png")
                xgb_dep_path = os.path.join(base_path, "xgboost_results_depth_km.png")
                with col3:
                    if os.path.exists(xgb_mag_path):
                        st.image(xgb_mag_path, caption="XGBoost Prediction: Magnitude", use_container_width=True)
                with col4:
                    if os.path.exists(xgb_dep_path):
                        st.image(xgb_dep_path, caption="XGBoost Prediction: Depth", use_container_width=True)

            #     st.subheader("🗺️ Coğrafi Dağılım")
            #     geo_img = os.path.join(base_path, "geographic_distribution.png")
            #     if os.path.exists(geo_img):
            #         st.image(geo_img, caption="Earthquake Geographic Distribution", use_container_width=True)
            # else:
            #     st.warning("📄 predictions.csv bulunamadı.")
        except Exception as e:
            st.error(f"AI Paneli yüklenemedi: {str(e)}")






# ==================================================




    # Proje Hakkında sekmesi
    elif selected_tab == "ℹ️ Proje Hakkında":
        st.title("🌍 Türkiye Deprem İzleme Sistemi")
        st.markdown("""
        <div style="background: linear-gradient(to right, #f8f9fa, #e9ecef);
                    padding: 25px;
                    border-radius: 15px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h3 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                Entegre Deprem Analiz Sistemi
            </h3>
            <p style="font-size: 1.1em;">Bu proje, Türkiye'deki deprem aktivitesini izlemek ve analiz etmek için en son veri bilimi ve yapay zeka teknolojilerini kullanarak entegre bir platform sağlamayı amaçlamaktadır.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background-color: #2c3e50; padding: 20px; border-radius: 10px; color: white;">
                <h4 style="color: #f1c40f;">📌 Ana Özellikler</h4>
                <ul style="font-size: 1em;">
                    <li>Dinamik Etkileşimli Haritalar</li>
                    <li>İleri Düzey İstatistiksel Analizler</li>
                    <li>Depremler için Zaman Sırası Görselleştirmesi</li>
                    <li>Otomatik Erken Uyarı Sistemi</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background-color: #e74c3c; padding: 20px; border-radius: 10px; color: white;">
                <h4 style="color: #f1c40f;">🛠️ Kullanılan Teknolojiler</h4>
                <ul style="font-size: 1em;">
                    <li>Python 3 + Streamlit</li>
                    <li>Pandas + NumPy için analiz</li>
                    <li>Folium + Plotly için görselleştirme</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center;color:#7f8c8d;font-size:0.9em;">
            Bu proje araştırma ve bilimsel amaçlar için geliştirilmiştir - © 2023 Tüm hakları saklıdır
        </div>
        """, unsafe_allow_html=True)
