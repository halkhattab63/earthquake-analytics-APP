import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os

_processed_df = None
# نستخدم هذا المتغير لتخزين نسخة من البيانات بعد تنظيفها ومعالجتها، حتى نستخدمها في دوال أخرى مثل الحفظ والرسم.

# الة لحساب المسافة بين نقطتين على سطح الأرض باستخدام صيغة Haversine (تأخذ خطوط الطول والعرض).
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between two points (in km)"""
    R = 6371  # Radius of Earth in km | نصف قطر الأرض بالكيلومترات
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2]) ## تحويل الدرجات إلى راديان 
     
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # حساب الفرق بين خط العرض وخط الطول بين النقطتين
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2 # حساب المسافة باستخدام صيغة هافيرسين
    #  صيغة هافرسين لقياس المسافة الزاوية.
    #  صيغة هافرسين هي صيغة رياضية تستخدم لحساب المسافة بين نقطتين على سطح الكرة الأرضية.
    #  تعتمد على الزاوية بين النقطتين بدلاً من المسافة الخطية.
    #  تستخدم هذه الصيغة في علم الجغرافيا والملاحة الجوية والبحرية.
    #  تعتمد على حساب الزاوية بين النقطتين باستخدام الدوال المثلثية.
    
    
    return 2 * R * np.arcsin(np.sqrt(a))  # تعطيك المسافة النهائية بالكيلومترات.




# تحميل البيانات
def load_data(filepath, use_smote=True):
    global _processed_df

    df = pd.read_csv(filepath) # يقرأ ملف CSV ويضعه في DataFrame.

    # Rename & clean basic columns
    # تحويل الوقت إلى نوع datetime، وحذف الصفوف المفقودة لأي من الأعمدة المهمة.
    df.rename(columns={'time': 'datetime'}, inplace=True)
    # نحاول تحويل القيم في العمود datetime إلى نوع تاريخ/زمن.
    # القيم التي لا يمكن تحويلها تُجعل NaT (مثل null).
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime', 'latitude', 'longitude', 'magnitude','depth_km','place'])# حذف الصفوف التي تحتوي على قيم مفقودة في الأعمدة المحددة

    # Add synthetic depth
    # توليد عمق عشوائي لكل زلزال (من 1 إلى 25 كم) – افتراض!
    df['depth_km'] = np.random.uniform(1, 10, size=len(df))

    # Time features
    df['month'] = df['datetime'].dt.month #Depremin meydana geldiği ay
    df['dayofweek'] = df['datetime'].dt.dayofweek #Depremin meydana geldiği gün ((0: Pazartesi))
    df['year'] = df['datetime'].dt.year
    df['hour'] = df['datetime'].dt.hour
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 20)).astype(int) # 1 إذا كان الزلزال في الليل، 0 إذا كان في النهار.

    # Physical features
    # طاقة الزلزال باستخدام صيغة تقريبية (بمقياس لوغاريتمي).
    df['energy'] = 10 ** (1.5 * df['magnitude'] + 4.8)#معادلة تقريبية لحساب الطاقة المنبعثة من الزلزال.
    df['severity_ratio'] = df['magnitude'] / (df['depth_km'] + 1) #شدة الزلزال بالنسبة للعمق.
    df['dist_fault'] = haversine_distance(df['latitude'], df['longitude'], 39.0, 34.0) # المسافة من نقطة الزلزال إلى خط الصدع (افتراضياً عند 39.0, 34.0).

    # Severity categories
    # تقسم الزلازل حسب magnitude إلى 5 فئات شدة.
    df['severity'] = pd.cut(
        df['magnitude'],
        bins=[0, 3, 4, 5, 6, 10],
        labels=['Hafif', 'Orta', 'Güçlü', 'Şiddetli', 'Felaket'],
        right=False
    )
    # حذف الزلازل اللي خارج التقسيم. تحويل التسمية إلى string.
    # تحويل الفئة إلى نص (مثل: "Şiddetli").
    df = df.dropna(subset=['severity'])
    df['severity_label'] = df['severity'].astype(str)
    
    
    label_mapping = {'Hafif': 0, 'Orta': 1, 'Güçlü': 2, 'Şiddetli': 3, 'Felaket': 4}
    df['severity_class'] = df['severity_label'].map(label_mapping)
    print(df[['severity', 'severity_label', 'severity_class']].head())
    
    
#       Feature selection
#       X: ميزات مدخلات النموذج.
#       y_class: التصنيف (للنموذج التصنيفي).
#       y_reg: التنبؤ (للنموذج التنبؤي).
    features = ['depth_km', 'magnitude', 'latitude', 'longitude', 'month', 'dayofweek',
                'year', 'is_night', 'energy', 'severity_ratio', 'dist_fault']
    X = df[features]
    y_class = df['severity_class']
    y_reg = df[['magnitude', 'depth_km']]



    # Standardization
    # توحيد كل عمود ليكون متوسطه = 0 والانحراف المعياري = 1.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# ********************************
    # حفظ الـ scaler
    os.makedirs("earthquake_risk_predictor/data", exist_ok=True)
    joblib.dump(scaler, "earthquake_risk_predictor/data/scaler.pkl")
    print("✅ Scaler saved to 'earthquake_risk_predictor/data/scaler.pkl'")
    
# *********************************
    
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
    
    
    
    # يتحقق من عدد العينات بكل فئة.
    # يستخدم SMOTE لإنشاء عينات صناعية في الفئات الأقل.
    # يعيد توليد y_reg لتتوافق مع توزيع الفئات الجديدة.
    # إذا كان هناك فئات أقل من 2، لا يستخدم SMOTE.
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
    # تقسيم إلى:
    # تدريب
    # اختبار
    # كل من التصنيف والتنبؤ
    # stratify=y_class يحافظ على نفس توزيع الفئات في المجموعتين.
    # test_size=0.3 يعني أن 30% من البيانات ستستخدم للاختبار.
    # random_state=42 لضمان تكرار النتائج.
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X_scaled, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )

    return X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg

# تحفظ النسخة المعالجة كـ CSV.
def export_processed_data():
    """Save the processed dataframe as CSV"""
    global _processed_df
    if _processed_df is not None:
        os.makedirs("data", exist_ok=True)
        _processed_df.to_csv("earthquake_risk_predictor/data/processed_earthquakes.csv", index=False)
        print("✅ Processed data saved to 'earthquake_risk_predictor/data/processed_earthquakes.csv'")


# ترسم وتحفظ توزيع الفئات (Bar chart).
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
        print("📊 Saved severity class distribution to 'earthquake_risk_predictor/data/severity_distribution.png'")