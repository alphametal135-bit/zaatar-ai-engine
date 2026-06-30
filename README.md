---
title: زعتر — Zaatar AI Engine
emoji: 🌿
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

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

---

صُنع بـ 🌿 — بدون أي API خارجي، كل الذكاء مبني محلياً.
