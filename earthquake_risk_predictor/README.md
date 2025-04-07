# Earthquake Risk Predictor

🔮 **Multi-Task Neural Network** to predict both:
- 🔢 Earthquake risk classification: `Low`, `Medium`, `High`
- 📈 Regression: Estimate `magnitude` and `depth_km`

---

## 🛠 Requirements
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
torch
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## 🚀 Run Training
```bash
python main.py
```
Saves model to `model.pt` and prints classification + regression results.

---

## 🔍 Run Inference on New Data
```python
import pandas as pd
from inference import predict

data = pd.DataFrame([{
  "depth_km": 10.0,
  "magnitude": 6.1,
  "latitude": 38.0,
  "longitude": 39.5
}])

cls, reg = predict(data)
print("Class:", cls)
print("Predicted (magnitude, depth_km):", reg)
```
