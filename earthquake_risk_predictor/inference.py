import torch
import pandas as pd
from src.model import EarthquakeModel
from sklearn.preprocessing import StandardScaler

# Load model and predict from raw input

def predict(input_data, model_path='model.pt'):
    model = EarthquakeModel(input_dim=4)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        class_out, reg_out = model(input_tensor)
        predicted_class = torch.argmax(class_out, dim=1).numpy()
        predicted_reg = reg_out.numpy()

    return predicted_class, predicted_reg
