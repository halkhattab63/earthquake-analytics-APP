# 📊 Smart Disaster Management for Earthquake Zones

## 🔍 Project Overview
This system analyzes past earthquake data, computes high-risk zones, determines safe evacuation routes, clusters emergency shelters, and presents all outputs on interactive maps.

---

## ✅ Data Sources
- Earthquake data: USGS API
- Road and Shelter data: OpenStreetMap via OSMnx

## 🧠 Components
- `KDE` for hotspot risk analysis
- `Dijkstra` for evacuation routing
- `KMeans` for shelter clustering
- `RandomForest` for risk classification

---

## 📂 Key Outputs
- Heatmap: `outputs/maps/kde_heatmap.png`
- Evacuation Map: `outputs/maps/evacuation_dashboard.html`
- Clustered shelters: `outputs/tables/shelter_clusters.*`
- Risk classification report: `outputs/tables/risk_classifier_report.txt`

---

## 🧪 Tests
Run:
```bash
pytest tests/test_integrity.py



# 🌍 Smart Disaster Management for Earthquake Zones | إدارة ذكية للكوارث في مناطق الزلازل

## Project Summary | ملخص المشروع
This project provides a smart, modular system to assist in disaster response and risk analysis in earthquake-prone areas, with a focus on Elazığ, Turkey. It integrates geospatial analysis, AI algorithms, and interactive mapping for decision support.

يقدم هذا المشروع نظاماً ذكياً متعدد الوظائف لدعم إدارة الكوارث وتحليل المخاطر في مناطق الزلازل، مع تركيز على مدينة إيلازغ التركية.

---

## Features | الوظائف الرئيسية

- ⚡ Fetch & clean earthquake data from USGS
- 🧬 Analyze risk zones using KDE heatmaps
- 🚪 Download road & shelter data from OSM
- 🌍 Build evacuation path using Dijkstra's algorithm
- 🏨 Cluster shelters using KMeans
- 🧠 Classify earthquake risk levels with RandomForest
- 🌐 Generate interactive evacuation maps with Folium
- 🔮 Run automated data integrity tests

---

## Directory Structure | تركيب الملفات
```
project yol /
├── src/                       # Analysis modules | وحدات التحليل
├── data/
│   ├── raw/                 # Raw input data | البيانات الأولية
│   └── cleaned/             # Cleaned data | بيانات منقاحة
├── outputs/
│   ├── maps/                # Visual outputs | خرائط
│   ├── tables/              # CSV/GeoJSON outputs | جداول
│   └── reports/             # Summary reports | تقارير
├── tests/                     # Pytest tests | اختبارات
└── main.py                    # Pipeline runner | ملف التشغيل
```

---

## How to Run | كيف تشغل النظام

```bash
pip install -r requirements.txt
python run_all.py
```

To test data validity: | لتشغيل الاختبارات:
```bash
pytest tests/test_integrity.py
```

---

## Credits | المساهمون
Developed by Hayan Alkhattab, 2025 | هيان الخطاب - 2025

---

## License | الترخيص
For academic use only | للاستخدام الأكاديمي فقط
