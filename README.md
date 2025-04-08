 taslak

# 🌍 Akıllı Deprem Yönetim ve Risk Analiz Platformu

Bu proje, Türkiye'deki geçmiş ve güncel deprem verilerini kullanarak **afet riski analizi**, **güvenli tahliye planlaması**, **sığınma alanı değerlendirmesi** ve **yapay zeka destekli sınıflandırma** gerçekleştiren entegre bir platformdur.

---

## 🚀 Özellikler

- 📡 Gerçek zamanlı veri çekimi (USGS API üzerinden)
- 🗺️ KDE ile riskli bölgelerin yoğunluk analizi
- 🧭 Dijkstra algoritması ile güvenli tahliye rotaları
- 🏕️ K-Means clustering ile sığınakların gruplandırılması
- 🧠 SVM, Random Forest ve Neural Network ile risk tahmini
- 📊 Streamlit + Folium + Plotly ile görselleştirme
- 📄 PDF / CSV / HTML formatlarında otomatik rapor üretimi

---

## 🧩 Sistem Mimarisi

```
project/
│
├── data/                    # Ham ve işlenmiş veri kaynakları
├── outputs/                 # Haritalar, raporlar ve sonuçlar
├── notebooks/              # Jupyter analiz defterleri
├── src/                    # Modüller
│   ├── fetch_usgs_data.py
│   ├── preprocess.py
│   ├── analyze_kde.py
│   ├── evacuation.py
│   ├── shelter_analysis.py
│   ├── risk_classifier.py
│   ├── visualize_risk_map.py
│   └── dashboard.py
└── model/                  # AI modeli ve değerlendirme
    ├── train.py
    ├── model.py
    ├── inference.py
    ├── evaluation.py
    └── preprocessing.py

project yol/
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── shapefiles/
│   └── geojson/
├── outputs/
│   ├── maps/
│   ├── tables/
│   └── reports/
├── src/
│   ├── fetch_usgs_data.py
│   ├── fetch_osm_data.py
│   ├── [preprocess.py](http://preprocess.py/)
│   ├── analyze_kde.py
│   ├── shelter_analysis.py
│   ├── [evacuation.py](http://evacuation.py/)
│   ├── risk_classifier.py
│   ├── generate_maps.py
│   └── [dashboard.py](http://dashboard.py/)
└── [main.py](http://main.py/) 
```

---

## 🧠 Kullanılan Algoritmalar ve Yöntemler

| Amaç | Yöntem / Algoritma |
|------|--------------------|
| Risk Bölgesi Belirleme | KDE (Kernel Density Estimation) |
| Tahliye Planlama | Dijkstra Algoritması |
| Sığınak Gruplama | K-Means Clustering |
| Risk Sınıflandırma | Random Forest, SVM, Neural Network |
| Veri Dengeleme | SMOTE |
| Özellik Seçimi | Derinlik, büyüklük, fay uzaklığı, enerji |

---

## ⚙️ Kurulum

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Streamlit arayüzünü başlat
streamlit run main.py
```

> Not: outputs/ klasöründe tüm görsel çıktılar ve raporlar saklanır.

---

## 📄 Üretilen Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `earthquake_report.pdf` | Otomatik oluşturulan analiz raporu |
| `predictions.csv` | AI modelinin tahmin çıktıları |
| `risk_zones_map.html` | KDE ile oluşturulan risk haritası |
| `evacuation_path.geojson` | Ana ve alternatif tahliye yolları |
| `shelter_clusters.csv` | Kümelenmiş sığınak koordinatları |

---

## 🧾 Proje Özeti

- Deprem verileri USGS API'den alınır
- KDE analizi ile yoğunluk hesaplanır
- Risk zonları çıkarılarak yollar ve sığınaklarla kesişimi analiz edilir
- Dijkstra algoritmasıyla güvenli yollar belirlenir
- AI modeli ile sınıflandırma (hafif / orta / felaket) ve büyüklük tahmini yapılır
- Sonuçlar interaktif haritalar ve PDF raporlarla sunulur

---

## 👤 Geliştirici Bilgileri

- **Ad Soyad:** Hayan Alkhattab  
- **Bölüm:**   
- **Dönem:** 2024-2025  
- **Danışman:** 

---

## 📜 Lisans

Bu proje sadece eğitim ve araştırma amacıyla geliştirilmiştir.  
Telif Hakkı © 2024 – Hayan Alkhattab 
