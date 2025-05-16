import torch
import torch.nn as nn
import torch.nn.functional as F

# torch: أساس مكتبة PyTorch، تُستخدم لتخزين tensors والتدريب.
# torch.nn: تحتوي على مكونات الشبكة العصبية مثل الطبقات، الدوال، الخسائر.
# torch.nn.functional: تحتوي على دوال تفعيل، دوال خسارة، وغيرها من الوظائف المفيدة.

# torch.nn: لبناء الشبكات العصبية مثل Linear, Sequential, ReLU.
# torch.nn.functional: للوصول إلى الدوال التفعيلية بشكل مباشر، مثل F.relu.


# تعريف كلاس يمثل نموذج مخصص للشبكة العصبية.
# يرث من nn.Module، وهي القاعدة لأي نموذج PyTorch.

# EarthquakeModel: نموذج شبكة عصبية لتصنيف شدة الزلازل والتنبؤ بالعمق والمقدار.
# يستخدم مكونات مثل Linear, BatchNorm1d, LeakyReLU, Dropout.

class EarthquakeModel(nn.Module):
    """
    Neural Network for Earthquake Severity Classification and Regression (Magnitude, Depth)
    Combines shared feature extraction, classification head, and regression head with output constraints.
    """
    
    # input_dim: عدد المدخلات (ميزات الزلزال).
    # num_classes: عدد الفئات (افتراضيًا 5).
    # super(...): لاستدعاء البناء الأساسي من nn.Module.
    # nn.Sequential: لتجميع عدة طبقات في تسلسل.
    # nn.Linear: طبقة خطية (Fully Connected Layer).
    def __init__(self, input_dim, num_classes=5):
        super(EarthquakeModel, self).__init__()


# Linear: طبقة خطية (تربط بين كل المدخلات والمخرجات).
# BatchNorm1d: توحيد القيم داخل كل batch (يساعد في الاستقرار).
# LeakyReLU: دالة تفعيلية، مثل ReLU لكن تسمح ببعض التدرج للقيم السالبة.
# Dropout: إطفاء عشوائي لـ 30% من الخلايا في كل طبقة لتقليل التخصيص الزائد (overfitting).
# المسار:
# من input_dim → 64 → 128 → 64
# والهدف: استخراج تمثيل مميز للمدخلات (الزلازل).

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )

        # Classification head (multi-class severity levels)
        # يأخذ الناتج من feature_extractor (حجم 64).
        # ويُخرج num_classes (عدد الفئات).
        # لا توجد دالة تفعيل هنا، لأنه سيتم استخدام CrossEntropyLoss لاحقًا الذي يدمج Softmax.
        self.class_head = nn.Linear(64, num_classes)

        # Regression head with Softplus activation to ensure positive outputs
        # يُخرج قيمتين: magnitude و depth.
        # Softplus: مثل ReLU لكن ناعمة، تضمن أن القيم الناتجة موجبة دائمًا – هذا مهم لأن العمق والمقدار لا يمكن أن يكونا سلبيين.
        self.reg_head = nn.Sequential(
            nn.Linear(64, 2),
            nn.Softplus()
        )
        
        
# x = self.feature_extractor(x)
# → يُطبق جميع الطبقات في سلسلة استخراج الميزات.
# class_out = self.class_head(x)
# → يمر عبر رأس التصنيف لإخراج logits للفئات.
# reg_out = self.reg_head(x)
# → يمر على رأس الانحدار ويُرجع [magnitude, depth].
# return
# → يُعيد قيمتين
# مخرجات تصنيفية (قبل softmax)
# ومخرجات تنبؤية (موجبة، حقيقية)
    def forward(self, x):
        x = self.feature_extractor(x)
        class_out = self.class_head(x)
        reg_out = self.reg_head(x)
        return class_out, reg_out
