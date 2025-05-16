# import torch
# import pandas as pd
# from src.model import EarthquakeModel
# from sklearn.preprocessing import StandardScaler

# # Load model and predict from raw input
# # دالة التنبؤ
# # تأخذ بيانات الإدخال (input_data) ومسار النموذج (model_path).
# # تقوم بتحميل النموذج، وتحويل البيانات إلى شكل tensor، ثم تقوم بالتنبؤ بالفئة والعمق.
# # تعيد الفئة المتوقعة والعمق المتوقع.
# # تستخدم مكتبة PyTorch لتحميل النموذج وتوقع النتائج.
# # تستخدم مكتبة pandas لتحويل البيانات إلى tensor.
# # model: نموذج الشبكة العصبية (EarthquakeModel).
# # input_data: بيانات الزلازل (مصفوفة من الميزات).
# # model_path: المسار إلى ملف النموذج المدرّب (.pt).
# # input_scaled: البيانات المدخلة بعد تحويلها باستخدام StandardScaler.
# # input_tensor: البيانات المدخلة على شكل tensor.
# def predict(input_data, model_path='model.pt'):
    
    
#     # إنشاء نسخة جديدة من نموذج EarthquakeModel.
#     # input_dim=4 → عدد الخصائص الداخلة (يفترض هنا فقط 4 خصائص!).
#     # ⚠️ تنبيه مهم: هذا الرقم لازم يتطابق مع عدد الخصائص المدخلة في التدريب، وإلا يحدث خطأ أبعاد.
#     model = EarthquakeModel(input_dim=4)
    
#     # تحميل حالة النموذج المدرب مسبقًا (الأوزان والمعاملات).
#     # torch.load(...): يحمّل الملف model.pt.
#     model.load_state_dict(torch.load(model_path))
    
#     # تفعيل وضع التقييم في PyTorch.
#     # يقوم بإيقاف Dropout و BatchNorm من التعلّم، ويجعلها ثابتة.
#     model.eval()


# # يتم توحيد بيانات الإدخال (نفس الشيء اللي تم أثناء التدريب).
# # ⚠️ لكن هنا خطأ شائع: المفروض نستخدم نفس scaler من التدريب، وليس تدريب واحد جديد على بيانات جديدة.
#     # هذا يضمن أن البيانات المدخلة تتناسب مع النموذج.
#     # StandardScaler: أداة لتحويل البيانات إلى توزيع طبيعي (mean=0, std=1).
#     scaler = StandardScaler()
#     input_scaled = scaler.fit_transform(input_data)
    
#     # نحول بيانات NumPy إلى Tensor (الصيغة التي يفهمها PyTorch).
#     input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    
# # إيقاف الحسابات الخاصة بالتدرج Gradient (تسريع العملية وتقليل استهلاك الذاكرة).
#     with torch.no_grad():
#         #  تشغيل النموذج والحصول على المخرجات:
# #         تمرير البيانات عبر النموذج.
# # المخرجات:
# # class_out: احتمالات الفئات (logits).
# # reg_out: قيم انحدار حقيقية (العمق والمقدار).
# # class_out: logits (قيمة غير محولة) للفئات.
# # reg_out: قيم انحدار (مثل العمق والمقدار).
#         class_out, reg_out = model(input_tensor)
        
#         #  تحليل النتائج:
# #         نأخذ الفئة التي لها أعلى احتمال (القيمة القصوى في المخرجات).
# # dim=1: عبر المحور الأفقي (لكل صف).
# # class_out: نستخدم softmax لتحويل logits إلى احتمالات.
# # reg_out: نستخدم numpy لتحويل القيم إلى مصفوفة NumPy.
#         predicted_class = torch.argmax(class_out, dim=1).numpy()
#         # استخراج قيم الانحدار (كمية الزلزال وعمقه).
#         predicted_reg = reg_out.numpy()


# # يُرجع:
# # مصفوفة predicted_class → الفئات المتوقعة.
# # مصفوفة predicted_reg → [magnitude, depth].

#     return predicted_class, predicted_reg




# أهم التحسينات:
# تحميل الـ StandardScaler الحقيقي من ملف محفوظ (scaler.pkl).

# دعم أي عدد ميزات تلقائيًا بدل input_dim=4.

# إضافة توثيق للدالة predict().

# التعامل مع الإدخال سواء كان DataFrame أو ndarray.

# تشغيل آمن للنموذج على CPU تلقائيًا.

import torch
import pandas as pd
import joblib
from earthquake_risk_predictor.src.PyTorch_predictor.model import EarthquakeModel  # يمكنك تغييره لـ SimpleNN أو أي موديل آخر
from sklearn.preprocessing import StandardScaler


def predict(input_data, model_path='model.pt', scaler_path='scaler.pkl'):
    """
    Perform inference using trained EarthquakeModel.

    Args:
        input_data (np.ndarray or pd.DataFrame): Input features.
        model_path (str): Path to the trained PyTorch model (.pt file).
        scaler_path (str): Path to the saved StandardScaler (.pkl file).

    Returns:
        tuple: (predicted_class: np.ndarray, predicted_reg: np.ndarray)
    """
    # Ensure input is DataFrame
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data)

    # Load scaler (trained during training phase)
    scaler = joblib.load(scaler_path)
    input_scaled = scaler.transform(input_data)

    # Load model
    input_dim = input_scaled.shape[1]
    model = EarthquakeModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Convert input to tensor
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Inference
    with torch.no_grad():
        class_out, reg_out = model(input_tensor)
        predicted_class = torch.argmax(class_out, dim=1).numpy()
        predicted_reg = reg_out.numpy()
# احفظ السكالر
    joblib.dump(scaler, "earthquake_risk_predictor/data/scaler.pkl")
    print("✅ Scaler saved to 'data/scaler.pkl'")
    return predicted_class, predicted_reg
