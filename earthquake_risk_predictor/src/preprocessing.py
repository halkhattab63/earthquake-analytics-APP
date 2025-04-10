import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os

_processed_df = None

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between two points (in km)"""
    R = 6371  # Radius of Earth in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_data(filepath, use_smote=True):
    global _processed_df

    df = pd.read_csv(filepath)

    # Rename & clean basic columns
    df.rename(columns={'time': 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime', 'latitude', 'longitude', 'magnitude'])

    # Add synthetic depth
    df['depth_km'] = np.random.uniform(1, 25, size=len(df))

    # Time features
    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['year'] = df['datetime'].dt.year
    df['hour'] = df['datetime'].dt.hour
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 20)).astype(int)

    # Physical features
    df['energy'] = 10 ** (1.5 * df['magnitude'] + 4.8)
    df['severity_ratio'] = df['magnitude'] / (df['depth_km'] + 1)
    df['dist_fault'] = haversine_distance(df['latitude'], df['longitude'], 39.0, 34.0)

    # Severity categories
    df['severity'] = pd.cut(
        df['magnitude'],
        bins=[0, 3, 4, 5, 6, 10],
        labels=['Hafif', 'Orta', 'Güçlü', 'Şiddetli', 'Felaket'],
        right=False
    )
    df = df.dropna(subset=['severity'])
    df['severity_label'] = df['severity'].astype(str)
    label_mapping = {'Hafif': 0, 'Orta': 1, 'Güçlü': 2, 'Şiddetli': 3, 'Felaket': 4}
    df['severity_class'] = df['severity_label'].map(label_mapping)

    # Feature selection
    features = ['depth_km', 'magnitude', 'latitude', 'longitude', 'month', 'dayofweek',
                'year', 'is_night', 'energy', 'severity_ratio', 'dist_fault']
    X = df[features]
    y_class = df['severity_class']
    y_reg = df[['magnitude', 'depth_km']]

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply SMOTE only if required
    # if use_smote:
    #     class_counts = y_class.value_counts()
    #     if all(class_counts >= 2):
    #         k_neighbors = max(1, min(class_counts.min() - 1, 5))
    #         smote = SMOTE(random_state=42, sampling_strategy='not majority', k_neighbors=k_neighbors)
    #         X_scaled, y_class = smote.fit_resample(X_scaled, y_class)

    #         # Rebalance y_reg (regression targets) with simple oversampling
    #         y_reg_resampled = []
    #         for label in sorted(y_class.unique()):
    #             label_indices = np.where(y_class == label)[0]
    #             if len(label_indices) > len(y_reg):
    #                 sample_rows = y_reg.sample(len(label_indices) - len(y_reg), replace=True, random_state=42)
    #                 y_reg_resampled.append(sample_rows)
    #         if y_reg_resampled:
    #             y_reg = pd.concat([y_reg] + y_reg_resampled, axis=0).reset_index(drop=True)
    if use_smote:
        class_counts = y_class.value_counts()
        if all(class_counts >= 2):
            k_neighbors = max(1, min(class_counts.min() - 1, 5))
            smote = SMOTE(random_state=42, sampling_strategy='not majority', k_neighbors=k_neighbors)
            X_scaled, y_class_resampled = smote.fit_resample(X_scaled, y_class)

            # الآن نعيد توليد y_reg بشكل مناسب بناءً على y_class_resampled
            y_reg_resampled = []

            label_mapping = y_class.reset_index(drop=True)  # هذا هو الأصلي
            for label in sorted(y_class_resampled.unique()):
                n_samples = sum(y_class_resampled == label)
                available = y_reg[y_class == label]
                if available.empty:
                    continue
                sampled = available.sample(n_samples, replace=True, random_state=42)
                y_reg_resampled.append(sampled)

            y_reg = pd.concat(y_reg_resampled, axis=0).reset_index(drop=True)
            y_class = y_class_resampled

    # Save processed dataframe for analysis
    _processed_df = df.copy()

    # Split dataset
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X_scaled, y_class, y_reg, test_size=0.3, random_state=42, stratify=y_class
    )

    return X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg


def export_processed_data():
    """Save the processed dataframe as CSV"""
    global _processed_df
    if _processed_df is not None:
        os.makedirs("data", exist_ok=True)
        _processed_df.to_csv("earthquake_risk_predictor/data/processed_earthquakes.csv", index=False)
        print("✅ Processed data saved to 'data/processed_earthquakes.csv'")


def plot_severity_distribution():
    """Plot and save class distribution"""
    global _processed_df
    if _processed_df is not None:
        plt.figure(figsize=(8, 5))
        _processed_df['severity_label'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title("Tehlike Sınıfı Dağılımı")
        plt.xlabel("Tehlike Seviyesi")
        plt.ylabel("Adet")
        plt.grid(True, axis='y')
        plt.tight_layout()
        os.makedirs("data", exist_ok=True)
        plt.savefig("earthquake_risk_predictor/data/severity_distribution.png")
        print("📊 Saved severity class distribution to 'data/severity_distribution.png'")