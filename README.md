# 🌿 زعتر v3 — Zaatar AI Engine

محرك ذكاء اصطناعي عربي مبني بالكامل من الصفر بـ **Python + NumPy**، بدون أي مكتبات تعلم آلي جاهزة (لا PyTorch ولا TensorFlow).

## ✨ المعمارية

| المكوّن | الوصف |
|---|---|
| **BPE Tokenizer** | Byte Pair Encoding — نفس أسلوب GPT للتعامل مع تصريفات الكلمات العربية |
| **Embedding Layer** | يحوّل كل token إلى متجه رقمي ذو معنى |
| **Transformer Block** | Multi-Head Self-Attention (4 رؤوس) + Positional Encoding + Feed-Forward + Layer Norm |
| **Bidirectional LSTM** | يقرأ الجملة من الاتجاهين لفهم أعمق للسياق |
| **Context Memory** | يدمج ذاكرة المحادثة السابقة مع كل رد جديد |
| **Classifier** | يصنّف النية النهائية ويختار رداً مناسباً |

كل الـ forward/backward pass (بما فيها BPTT للـ LSTM وbackprop الكامل عبر الـ Attention) مكتوبة يدوياً بـ NumPy.

## 🚀 التشغيل محلياً

```bash
pip install -r requirements.txt
python app.py
```

سيفتح السيرفر على `http://localhost:5000` — افتحه في المتصفح وابدأ المحادثة مباشرة. أول تشغيل سيقوم بتدريب النموذج تلقائياً (~250 epoch) ثم يحفظ الأوزان في `zaatar_v3.pkl`.

## 🌐 النشر (Deploy)

يدعم أي منصة تشغّل تطبيقات Flask (Render, Railway, Fly.io, إلخ). المتغير `PORT` يُقرأ تلقائياً من البيئة.

مثال لـ Render:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python app.py`

## 📡 REST API

| Method | Endpoint | الوصف |
|---|---|---|
| GET | `/` | واجهة الدردشة |
| GET | `/status` | حالة النموذج والمعمارية |
| POST | `/chat` | `{"message": "مرحبا"}` → رد زعتر |
| POST | `/learn` | `{"text": "...", "intent": 0}` تعلم فوري |
| GET | `/memory` | سجل المحادثة الحالي |
| DELETE | `/memory` | مسح الذاكرة |
| POST | `/attention` | خريطة انتباه الجملة |
| POST | `/save` | حفظ الأوزان الحالية |

## 📁 هيكل المشروع

```
.
├── app.py                      # المحرك الكامل + Flask API
├── templates/
│   └── zaatar_chat.html        # واجهة الدردشة (تُخدَّم من نفس السيرفر)
├── requirements.txt
└── README.md
```

## 🧠 النوايا الافتراضية

النموذج مدرَّب افتراضياً على 4 نوايا: `greeting` (ترحيب)، `goodbye` (وداع)، `identity` (هوية)، `thanks` (شكر). يمكن توسيعها بسهولة عبر `/learn` أو بتعديل `X_train`/`y_train` داخل `app.py`.

---

صُنع بـ 🌿 — بدون أي API خارجي، كل الذكاء مبني محلياً.
