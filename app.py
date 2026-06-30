"""
╔══════════════════════════════════════════════════════════════════════╗
║           AI_ENGINE Zaatar v3.0 — المحرك الكامل                    ║
║                                                                      ║
║  ✅ 1. LSTM          (بوابات Forget/Input/Output)                   ║
║  ✅ 2. Bidirectional RNN  (قراءة الجملة من الاتجاهين)              ║
║  ✅ 3. Multi-Head Attention  (12 رأس موازي)                         ║
║  ✅ 4. Transformer Block  (كامل مع Positional Encoding)             ║
║  ✅ 5. BPE Tokenizer  (Byte Pair Encoding للعربية)                  ║
║  ✅ 6. Flask REST API  (ربط Python بالواجهة)                        ║
║  ✅ 7. Model Save/Load  (حفظ وتحميل الأوزان)                       ║
╚══════════════════════════════════════════════════════════════════════╝

التشغيل:
    pip install flask flask-cors numpy
    python AI_ENGINE_Zaatar_v3.py

الـ API:
    POST /chat      → {"message": "مرحبا"}
    POST /learn     → {"text": "جملة", "intent": 0}
    GET  /memory    → سياق المحادثة
    GET  /status    → حالة النموذج
"""

import numpy as np
import json
import os
import sys
import random
import pickle
import re
from collections import Counter

# ══════════════════════════════════════════════════
# ACTIVATIONS
# ══════════════════════════════════════════════════

class ReLU:
    def forward(self, z):
        self.input = z
        return np.maximum(0, z)
    def backward(self, grad):
        return grad * (self.input > 0)

class Sigmoid:
    def forward(self, z):
        self.output = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return self.output
    def backward(self, grad):
        return grad * self.output * (1 - self.output)

class Tanh:
    def forward(self, z):
        self.output = np.tanh(z)
        return self.output
    def backward(self, grad):
        return grad * (1 - self.output ** 2)

def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


# ══════════════════════════════════════════════════
# LAYERS
# ══════════════════════════════════════════════════

class Layer:
    def __init__(self, n_inputs, n_neurons, activation=None):
        self.weights   = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases    = np.zeros((1, n_neurons))
        self.activation = activation
        self.d_weights = None
        self.d_biases  = None

    def forward(self, x):
        self.input = x
        z = np.dot(x, self.weights) + self.biases
        self.output = self.activation.forward(z) if self.activation else z
        return self.output

    def backward(self, grad):
        if self.activation:
            grad = self.activation.backward(grad)
        self.d_weights = np.dot(self.input.T, grad)
        self.d_biases  = np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.weights.T)


class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        # تهيئة بمقياس مناسب يوازن مع Positional Encoding (مدى ±1)
        # بدلاً من 0.01 (صغير جداً وتطغى عليه الإشارة الموضعية)
        self.embeddings   = np.random.randn(vocab_size, embedding_dim) * (1.0 / np.sqrt(embedding_dim))
        self.embedding_dim = embedding_dim
        self.vocab_size   = vocab_size
        self.d_embeddings = None

    def forward(self, token_ids):
        self.token_ids = token_ids
        return self.embeddings[token_ids]

    def backward(self, grad):
        self.d_embeddings = np.zeros_like(self.embeddings)
        np.add.at(self.d_embeddings, self.token_ids, grad)

    def update(self, lr=0.001):
        if self.d_embeddings is not None:
            self.embeddings -= lr * np.clip(self.d_embeddings, -5, 5)


# ══════════════════════════════════════════════════
# ✅ 1. LSTM LAYER
#    بوابات: Forget · Input · Output · Cell State
# ══════════════════════════════════════════════════

class LSTMLayer:
    """
    Long Short-Term Memory — يحل مشكلة نسيان المدى البعيد في RNN.

    البوابات الأربع:
    ┌─────────────┬──────────────────────────────────────────┐
    │ Forget Gate │ كم من الذاكرة القديمة نحتفظ بها؟       │
    │ Input Gate  │ ما الجديد الذي نضيفه للذاكرة؟          │
    │ Cell Gate   │ المحتوى الجديد المرشح للذاكرة           │
    │ Output Gate │ ماذا نُخرج من الذاكرة الآن؟             │
    └─────────────┴──────────────────────────────────────────┘

    المعادلات:
        f(t) = σ(W_f · [h(t-1), x(t)] + b_f)   ← Forget
        i(t) = σ(W_i · [h(t-1), x(t)] + b_i)   ← Input
        g(t) = tanh(W_g · [h(t-1), x(t)] + b_g) ← Cell candidate
        o(t) = σ(W_o · [h(t-1), x(t)] + b_o)   ← Output
        c(t) = f(t) ⊙ c(t-1) + i(t) ⊙ g(t)     ← Cell State
        h(t) = o(t) ⊙ tanh(c(t))                ← Hidden State
    """
    def __init__(self, input_size, hidden_size):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        combined = input_size + hidden_size

        # تهيئة الأوزان الأربع (Forget, Input, Cell, Output)
        scale = np.sqrt(2.0 / combined)
        self.W_f = np.random.randn(combined, hidden_size) * scale
        self.W_i = np.random.randn(combined, hidden_size) * scale
        self.W_g = np.random.randn(combined, hidden_size) * scale
        self.W_o = np.random.randn(combined, hidden_size) * scale

        # Forget gate bias = 1 (ممارسة شائعة لتسريع التعلم)
        self.b_f = np.ones((1, hidden_size))
        self.b_i = np.zeros((1, hidden_size))
        self.b_g = np.zeros((1, hidden_size))
        self.b_o = np.zeros((1, hidden_size))

        self._init_grads()

    def _init_grads(self):
        for attr in ['W_f','W_i','W_g','W_o','b_f','b_i','b_g','b_o']:
            setattr(self, 'd_'+attr, np.zeros_like(getattr(self, attr)))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, x_sequence):
        """
        x_sequence: (batch, seq_len, input_size)
        returns: outputs (batch, seq_len, hidden_size), last_h, last_c
        """
        batch, seq_len, _ = x_sequence.shape

        h = np.zeros((batch, self.hidden_size))
        c = np.zeros((batch, self.hidden_size))

        self.cache = []
        outputs    = []

        for t in range(seq_len):
            x_t      = x_sequence[:, t, :]           # (batch, input_size)
            combined = np.concatenate([h, x_t], axis=1)  # (batch, combined)

            f = self._sigmoid(np.dot(combined, self.W_f) + self.b_f)
            i = self._sigmoid(np.dot(combined, self.W_i) + self.b_i)
            g = np.tanh(np.dot(combined,       self.W_g) + self.b_g)
            o = self._sigmoid(np.dot(combined, self.W_o) + self.b_o)

            c_new = f * c + i * g
            h_new = o * np.tanh(c_new)

            self.cache.append((x_t, h, c, f, i, g, o, c_new, combined))

            h = h_new
            c = c_new
            outputs.append(h.copy())

        self.last_h = h
        self.last_c = c
        return np.stack(outputs, axis=1), h, c  # (batch, seq, hidden), ...

    def backward(self, d_outputs, d_h_next=None, d_c_next=None):
        """BPTT للـ LSTM"""
        batch = d_outputs.shape[0]
        if d_h_next is None: d_h_next = np.zeros((batch, self.hidden_size))
        if d_c_next is None: d_c_next = np.zeros((batch, self.hidden_size))

        dx_list = []

        for t in reversed(range(len(self.cache))):
            x_t, h_prev, c_prev, f, i, g, o, c_new, combined = self.cache[t]

            dh = d_outputs[:, t, :] + d_h_next
            dc = d_c_next + dh * o * (1 - np.tanh(c_new)**2)

            df = dc * c_prev
            di = dc * g
            dg = dc * i
            do = dh * np.tanh(c_new)

            # تدرجات بوابات
            df_raw = df * f * (1 - f)
            di_raw = di * i * (1 - i)
            dg_raw = dg * (1 - g**2)
            do_raw = do * o * (1 - o)

            # تدرجات الأوزان
            self.d_W_f += np.dot(combined.T, df_raw)
            self.d_W_i += np.dot(combined.T, di_raw)
            self.d_W_g += np.dot(combined.T, dg_raw)
            self.d_W_o += np.dot(combined.T, do_raw)
            self.d_b_f += np.sum(df_raw, axis=0, keepdims=True)
            self.d_b_i += np.sum(di_raw, axis=0, keepdims=True)
            self.d_b_g += np.sum(dg_raw, axis=0, keepdims=True)
            self.d_b_o += np.sum(do_raw, axis=0, keepdims=True)

            d_combined = (np.dot(df_raw, self.W_f.T) +
                         np.dot(di_raw, self.W_i.T) +
                         np.dot(dg_raw, self.W_g.T) +
                         np.dot(do_raw, self.W_o.T))

            d_h_next = d_combined[:, :self.hidden_size]
            dx       = d_combined[:, self.hidden_size:]
            d_c_next = dc * f
            dx_list.append(dx)

        return np.stack(list(reversed(dx_list)), axis=1)

    def update(self, lr=0.001, clip=5.0):
        for attr in ['W_f','W_i','W_g','W_o','b_f','b_i','b_g','b_o']:
            grad  = np.clip(getattr(self, 'd_'+attr), -clip, clip)
            param = getattr(self, attr)
            param -= lr * grad
        self._init_grads()

    def __repr__(self):
        return f"LSTMLayer(input={self.input_size}, hidden={self.hidden_size})"


# ══════════════════════════════════════════════════
# ✅ 2. BIDIRECTIONAL RNN (يستخدم LSTM داخلياً)
# ══════════════════════════════════════════════════

class BidirectionalLSTM:
    """
    يشغّل LSTM في اتجاهين ويدمج النتيجتين:

    Forward:  x₁ → x₂ → x₃ → x₄  (من اليسار لليمين)
    Backward: x₄ → x₃ → x₂ → x₁  (من اليمين لليسار)
    Output:   [h_fwd; h_bwd]  (تجميع من الاتجاهين)

    مفيد جداً للعربية:
    "لا أحب القهوة" ← كلمة "لا" تؤثر على "أحب" للأمام والخلف
    """
    def __init__(self, input_size, hidden_size):
        self.fwd = LSTMLayer(input_size, hidden_size)
        self.bwd = LSTMLayer(input_size, hidden_size)
        self.hidden_size    = hidden_size
        self.output_size    = hidden_size * 2  # من الاتجاهين

    def forward(self, x_sequence):
        """
        x_sequence: (batch, seq_len, input_size)
        returns: (batch, hidden*2) — آخر حالة من كل اتجاه
        """
        # الاتجاه الأمامي
        _, h_fwd, _ = self.fwd.forward(x_sequence)

        # الاتجاه العكسي (نعكس التسلسل)
        x_rev = x_sequence[:, ::-1, :]
        _, h_bwd, _ = self.bwd.forward(x_rev)

        # دمج الاتجاهين
        self.h_fwd = h_fwd
        self.h_bwd = h_bwd
        return np.concatenate([h_fwd, h_bwd], axis=1)  # (batch, hidden*2)

    def backward(self, grad):
        """
        grad: (batch, hidden*2) — تدرج بخصوص آخر hidden state من الاتجاهين فقط
        (لأن forward يستخدم h الأخير فقط من كل اتجاه)
        """
        half  = self.hidden_size
        batch = grad.shape[0]
        seq_len = len(self.fwd.cache)

        # نبني تسلسل أصفار بطول الجملة، ونضع التدرج عند آخر خطوة فقط
        zero_seq = np.zeros((batch, seq_len, half))

        dx_fwd = self.fwd.backward(zero_seq, d_h_next=grad[:, :half])
        # للاتجاه العكسي: forward قرأ x_rev، لذلك آخر خطوة فيه هي أول كلمة فعلياً
        dx_bwd_rev = self.bwd.backward(zero_seq, d_h_next=grad[:, half:])

        # نعكس dx_bwd مرة أخرى لمطابقة الترتيب الأصلي
        dx_bwd = dx_bwd_rev[:, ::-1, :]

        # مجموع التدرجات القادمة للمدخل المشترك (embedding/transformer output)
        return dx_fwd + dx_bwd

    def update(self, lr=0.001):
        self.fwd.update(lr)
        self.bwd.update(lr)

    def __repr__(self):
        return (f"BiLSTM(hidden_each={self.hidden_size}, "
                f"output={self.output_size})")


# ══════════════════════════════════════════════════
# ✅ 3. MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════

class MultiHeadAttention:
    """
    Multi-Head Attention — قلب معمارية Transformer.

    بدل رأس انتباه واحد، نشغّل H رؤوس موازية:
    كل رأس يتعلم الانتباه لجانب مختلف من الجملة:
    - رأس 1: العلاقات النحوية
    - رأس 2: المترادفات
    - رأس 3: النفي والتأكيد
    - ... إلخ

    المعادلة:
        head_i = Attention(Q·Wq_i, K·Wk_i, V·Wv_i)
        MultiHead = Concat(head_1,...,head_H) · Wo
    """
    def __init__(self, d_model, num_heads=4):
        assert d_model % num_heads == 0, "d_model يجب أن يقسم على num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads  # حجم كل رأس

        scale = np.sqrt(2.0 / d_model)
        # مصفوفة لكل رأس (نجمعها في مصفوفة واحدة للكفاءة)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

        self.scale_factor = np.sqrt(self.d_k)

    def _split_heads(self, x):
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        batch, seq, _ = x.shape
        x = x.reshape(batch, seq, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        """(batch, heads, seq, d_k) → (batch, seq, d_model)"""
        batch, _, seq, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch, seq, self.d_model)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """
        self.x = x
        batch, seq, _ = x.shape

        # إسقاط Q, K, V
        Q = np.dot(x, self.W_Q)  # (batch, seq, d_model)
        K = np.dot(x, self.W_K)
        V = np.dot(x, self.W_V)
        self.V = V

        # تقسيم على الرؤوس
        Q_h = self._split_heads(Q)  # (batch, heads, seq, d_k)
        K_h = self._split_heads(K)
        V_h = self._split_heads(V)

        # Scaled Dot-Product Attention لكل رأس
        scores = np.matmul(Q_h, K_h.transpose(0, 1, 3, 2)) / self.scale_factor
        self.attn_weights = softmax(scores)  # (batch, heads, seq, seq)

        # تجميع القيم
        context = np.matmul(self.attn_weights, V_h)  # (batch, heads, seq, d_k)
        context = self._merge_heads(context)           # (batch, seq, d_model)

        # الإسقاط النهائي
        self.context = context
        output = np.dot(context, self.W_O)
        return output

    def backward(self, d_output):
        """
        Backpropagation كامل عبر Multi-Head Attention.
        d_output: (batch, seq, d_model)
        يعيد: dx (batch, seq, d_model)
        """
        batch, seq, _ = d_output.shape

        # ── W_O ──
        self.d_W_O = np.einsum('bsd,bsc->dc', self.context, d_output)
        d_context  = np.dot(d_output, self.W_O.T)  # (batch, seq, d_model)

        # ── split heads للتدرج ──
        d_context_h = self._split_heads(d_context)  # (batch, heads, seq, d_k)
        V_h = self._split_heads(self.V)
        K_h = self._split_heads(np.dot(self.x, self.W_K))
        Q_h = self._split_heads(np.dot(self.x, self.W_Q))

        # ── تدرج V عبر attn_weights·V ──
        d_V_h = np.matmul(self.attn_weights.transpose(0,1,3,2), d_context_h)  # (b,h,seq,d_k)
        d_attn = np.matmul(d_context_h, V_h.transpose(0,1,3,2))               # (b,h,seq,seq)

        # ── تدرج softmax ──
        s = self.attn_weights
        d_scores = s * (d_attn - np.sum(d_attn * s, axis=-1, keepdims=True))
        d_scores = d_scores / self.scale_factor

        # ── تدرج Q,K من scores = Q·Kᵀ ──
        d_Q_h = np.matmul(d_scores, K_h)               # (b,h,seq,d_k)
        d_K_h = np.matmul(d_scores.transpose(0,1,3,2), Q_h)

        d_Q = self._merge_heads(d_Q_h)  # (batch, seq, d_model)
        d_K = self._merge_heads(d_K_h)
        d_V = self._merge_heads(d_V_h)

        # ── تدرجات أوزان الإسقاط ──
        self.d_W_Q = np.einsum('bsd,bsc->dc', self.x, d_Q)
        self.d_W_K = np.einsum('bsd,bsc->dc', self.x, d_K)
        self.d_W_V = np.einsum('bsd,bsc->dc', self.x, d_V)

        # ── تدرج المدخل x (من الفروع الثلاثة Q,K,V) ──
        dx = (np.dot(d_Q, self.W_Q.T) +
              np.dot(d_K, self.W_K.T) +
              np.dot(d_V, self.W_V.T))
        return dx

    def update(self, lr=0.001, clip=5.0):
        for name in ['W_Q', 'W_K', 'W_V', 'W_O']:
            grad  = np.clip(getattr(self, 'd_' + name), -clip, clip)
            param = getattr(self, name)
            param -= lr * grad

    def get_attention_map(self):
        """يعيد خريطة الانتباه لكل رأس — للتفسير"""
        return self.attn_weights  # (batch, heads, seq, seq)

    def __repr__(self):
        return (f"MultiHeadAttention(d_model={self.d_model}, "
                f"heads={self.num_heads}, d_k={self.d_k})")


# ══════════════════════════════════════════════════
# ✅ 4. TRANSFORMER BLOCK
# ══════════════════════════════════════════════════

class PositionalEncoding:
    """
    يضيف معلومات الموضع لكل كلمة.
    بدونه، الجملة "أنا أحب القهوة" مطابقة لـ "القهوة أحب أنا".

    المعادلة (Vaswani et al. 2017):
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, max_len=50, d_model=32):
        pe  = np.zeros((max_len, d_model))
        pos = np.arange(max_len).reshape(-1, 1)
        div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:d_model//2])

        self.pe = pe  # (max_len, d_model)

    def forward(self, x):
        """x: (batch, seq, d_model)"""
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]


class LayerNorm:
    """Layer Normalization — يستقر التدريب"""
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta  = np.zeros(d_model)
        self.eps   = eps

    def forward(self, x):
        self.x    = x
        self.mean = x.mean(axis=-1, keepdims=True)
        self.std  = x.std(axis=-1, keepdims=True)
        self.xn   = (x - self.mean) / (self.std + self.eps)
        return self.gamma * self.xn + self.beta

    def backward(self, d_out):
        """تدرج Layer Norm — صيغة مبسطة وفعالة عددياً"""
        self.d_gamma = np.sum(d_out * self.xn, axis=tuple(range(d_out.ndim - 1)))
        self.d_beta  = np.sum(d_out, axis=tuple(range(d_out.ndim - 1)))

        N = self.x.shape[-1]
        d_xn = d_out * self.gamma
        std_inv = 1.0 / (self.std + self.eps)

        dx = (1.0 / N) * std_inv * (
            N * d_xn
            - np.sum(d_xn, axis=-1, keepdims=True)
            - self.xn * np.sum(d_xn * self.xn, axis=-1, keepdims=True)
        )
        return dx

    def update(self, lr=0.001):
        self.gamma -= lr * np.clip(self.d_gamma, -5, 5)
        self.beta  -= lr * np.clip(self.d_beta, -5, 5)


class FeedForward:
    """
    طبقة Feed-Forward داخل Transformer:
        FFN(x) = max(0, x·W₁+b₁)·W₂+b₂
    """
    def __init__(self, d_model, d_ff=None):
        d_ff = d_ff or d_model * 4
        self.W1 = np.random.randn(d_model, d_ff)  * np.sqrt(2.0/d_model)
        self.b1 = np.zeros((1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model)  * np.sqrt(2.0/d_ff)
        self.b2 = np.zeros((1, d_model))

    def forward(self, x):
        self.x  = x
        self.h  = np.maximum(0, np.dot(x, self.W1) + self.b1)  # ReLU
        return np.dot(self.h, self.W2) + self.b2

    def backward(self, d_out):
        """d_out: (batch, seq, d_model)"""
        self.d_W2 = np.einsum('bsf,bsd->fd', self.h, d_out)
        self.d_b2 = np.sum(d_out, axis=(0, 1), keepdims=True).reshape(1, -1)

        d_h = np.dot(d_out, self.W2.T)
        d_h_relu = d_h * (self.h > 0)

        self.d_W1 = np.einsum('bsd,bsf->df', self.x, d_h_relu)
        self.d_b1 = np.sum(d_h_relu, axis=(0, 1), keepdims=True).reshape(1, -1)

        dx = np.dot(d_h_relu, self.W1.T)
        return dx

    def update(self, lr=0.001, clip=5.0):
        for name in ['W1', 'b1', 'W2', 'b2']:
            grad  = np.clip(getattr(self, 'd_' + name), -clip, clip)
            param = getattr(self, name)
            param -= lr * grad


class TransformerBlock:
    """
    كتلة Transformer كاملة:

    x → [Multi-Head Attention] → Add & Norm
      → [Feed Forward]          → Add & Norm
      → output

    هذه الكتلة تتكرر N مرات في GPT/BERT/Claude
    """
    def __init__(self, d_model, num_heads=4):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1     = LayerNorm(d_model)
        self.norm2     = LayerNorm(d_model)
        self.ff        = FeedForward(d_model)
        self.pos_enc   = PositionalEncoding(d_model=d_model)

    def forward(self, x):
        # إضافة Positional Encoding
        x_pos = self.pos_enc.forward(x)
        self.x_pos = x_pos

        # Multi-Head Attention + Residual
        attn_out = self.attention.forward(x_pos)
        self.pre_norm1 = x_pos + attn_out
        x1 = self.norm1.forward(self.pre_norm1)  # Add & Norm
        self.x1 = x1

        # Feed Forward + Residual
        ff_out = self.ff.forward(x1)
        self.pre_norm2 = x1 + ff_out
        x2 = self.norm2.forward(self.pre_norm2)       # Add & Norm

        return x2  # (batch, seq, d_model)

    def backward(self, d_out):
        """
        Backprop كامل عبر كتلة الـ Transformer بأكملها:
        Norm2 → (residual: FF + x1) → Norm1 → (residual: Attn + x_pos)
        """
        # ── Norm2 ──
        d_pre_norm2 = self.norm2.backward(d_out)

        # ── Residual split: pre_norm2 = x1 + ff_out ──
        d_x1_from_res = d_pre_norm2          # مسار residual المباشر
        d_ff_out      = d_pre_norm2          # مسار عبر FeedForward

        # ── FeedForward ──
        d_x1_from_ff = self.ff.backward(d_ff_out)

        d_x1 = d_x1_from_res + d_x1_from_ff

        # ── Norm1 ──
        d_pre_norm1 = self.norm1.backward(d_x1)

        # ── Residual split: pre_norm1 = x_pos + attn_out ──
        d_x_pos_from_res = d_pre_norm1
        d_attn_out        = d_pre_norm1

        # ── Multi-Head Attention ──
        d_x_pos_from_attn = self.attention.backward(d_attn_out)

        d_x_pos = d_x_pos_from_res + d_x_pos_from_attn

        # Positional encoding ثابت (لا أوزان قابلة للتعلم) — التدرج يمر مباشرة
        dx = d_x_pos
        return dx

    def update(self, lr=0.001):
        self.attention.update(lr)
        self.norm1.update(lr)
        self.norm2.update(lr)
        self.ff.update(lr)

    def get_attention_map(self):
        return self.attention.get_attention_map()

    def __repr__(self):
        return f"TransformerBlock(d_model={self.ff.W1.shape[0]}, heads={self.attention.num_heads})"


# ══════════════════════════════════════════════════
# ✅ 5. BPE TOKENIZER (Byte Pair Encoding)
# ══════════════════════════════════════════════════

class BPETokenizer:
    """
    Byte Pair Encoding — أساس GPT-2/3/4 والنماذج الحديثة.

    المشكلة التي يحلها:
    - "سيارتي" و"سيارتك" و"سيارته" → كلمات مختلفة في القاموس العادي
    - BPE يتعلم أن "سيارة" + "تي/تك/ته" وحدات منفصلة

    الخوارزمية:
    1. ابدأ بحروف فردية كـ tokens
    2. اعثر على أكثر زوج متجاور تكراراً
    3. ادمجهما في token جديد
    4. كرر N مرة

    مثال:
        "مرحبا" → ['م','ر','ح','ب','ا'] → ['مر','ح','ب','ا'] → ['مرح','با'] ...
    """
    def __init__(self, vocab_size=500):
        self.target_vocab_size = vocab_size
        self.vocab   = {}
        self.merges  = []
        self.special = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

    def _get_pairs(self, word_freqs):
        """يحسب تكرار كل زوج متجاور"""
        pairs = Counter()
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def _merge_pair(self, pair, word_freqs):
        """يدمج الزوج الأكثر تكراراً"""
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    def fit(self, texts):
        """تدريب BPE على النصوص"""
        # تقسيم كل كلمة إلى حروف + علامة نهاية
        word_freqs = Counter()
        for text in texts:
            for word in text.split():
                # نضيف مسافة بين كل حرفين لجعل الدمج ممكناً
                spaced = ' '.join(list(word)) + ' </w>'
                word_freqs[spaced] += 1

        # بناء القاموس الأولي
        vocab = dict(self.special)
        all_chars = set()
        for word in word_freqs:
            for ch in word.split():
                all_chars.add(ch)

        for ch in sorted(all_chars):
            if ch not in vocab:
                vocab[ch] = len(vocab)

        # دورات الدمج
        num_merges = self.target_vocab_size - len(vocab)
        for step in range(max(0, num_merges)):
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            word_freqs = self._merge_pair(best_pair, word_freqs)
            new_token = ''.join(best_pair)
            if new_token not in vocab:
                vocab[new_token] = len(vocab)

        self.vocab         = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}

    def _tokenize_word(self, word):
        """يطبق قواعد الدمج على كلمة واحدة"""
        if not word:
            return []
        symbols = list(word) + ['</w>']
        for pair in self.merges:
            i = 0
            new_symbols = []
            while i < len(symbols):
                if (i < len(symbols) - 1 and
                        symbols[i] == pair[0] and
                        symbols[i+1] == pair[1]):
                    new_symbols.append(''.join(pair))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text, max_len=16, pad=True):
        """تحويل النص إلى قائمة أرقام"""
        ids = [self.special["<BOS>"]]
        for word in text.split():
            tokens = self._tokenize_word(word)
            for t in tokens:
                ids.append(self.vocab.get(t, self.special["<UNK>"]))
        ids.append(self.special["<EOS>"])

        if pad:
            if len(ids) < max_len:
                ids += [self.special["<PAD>"]] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
        return ids

    def decode(self, ids):
        """تحويل أرقام إلى نص"""
        tokens = [self.inverse_vocab.get(i, "?") for i in ids
                  if i not in self.special.values()]
        return ''.join(tokens).replace('</w>', ' ').strip()

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __repr__(self):
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"


# ══════════════════════════════════════════════════
# OPTIMIZER — Adam
# ══════════════════════════════════════════════════

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.cache = {}

    def step(self, param, grad):
        key = id(param)
        if key not in self.cache:
            self.cache[key] = {
                'm': np.zeros_like(param),
                'v': np.zeros_like(param)
            }
        self.t += 1
        c    = self.cache[key]
        c['m'] = self.beta1 * c['m'] + (1 - self.beta1) * grad
        c['v'] = self.beta2 * c['v'] + (1 - self.beta2) * grad**2
        m_hat  = c['m'] / (1 - self.beta1**self.t)
        v_hat  = c['v'] / (1 - self.beta2**self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param


# ══════════════════════════════════════════════════
# CONTEXT MEMORY
# ══════════════════════════════════════════════════

class ConversationMemory:
    def __init__(self, max_size=10):
        self.history  = []
        self.max_size = max_size

    def add(self, user, bot, intent, confidence):
        self.history.append({
            "user": user, "bot": bot,
            "intent": intent, "confidence": float(confidence)
        })
        if len(self.history) > self.max_size:
            self.history.pop(0)

    def context_vector(self, dim=16):
        ctx = np.zeros(dim)
        if not self.history:
            return ctx
        ctx[0] = len(self.history) / self.max_size
        ctx[1] = np.mean([h['confidence'] for h in self.history])
        for i, h in enumerate(self.history[-3:]):
            slot = 2 + i * 4
            intent = min(int(h['intent']), 3)
            if slot + intent < dim:
                ctx[slot + intent] = 1.0
        return ctx

    def to_dict(self):
        return self.history[-5:]


# ══════════════════════════════════════════════════
# ZAATAR v3 — MAIN MODEL
# ══════════════════════════════════════════════════

class ZaatarV3:
    """
    المحرك الكامل يجمع:
    BPETokenizer → EmbeddingLayer → TransformerBlock
    → [BiLSTM branch] → دمج → Classifier
    """
    SEQ_LEN    = 8
    EMBED_DIM  = 32
    HIDDEN     = 32
    D_MODEL    = 32
    NUM_HEADS  = 4
    CONTEXT_DIM = 16

    def __init__(self, intents, responses):
        self.intents   = intents
        self.responses = responses
        self.memory    = ConversationMemory()
        self.tokenizer = BPETokenizer(vocab_size=300)
        self.trained   = False

        self.embed       = None
        self.transformer = None
        self.bilstm      = None
        self.classifier  = None
        self.optimizer   = Adam(lr=0.02)

        self.X_data = []
        self.y_data = []

    def _build(self):
        num_classes = len(self.intents)
        vocab_size  = self.tokenizer.vocab_size

        self.embed       = EmbeddingLayer(vocab_size, self.EMBED_DIM)
        self.transformer = TransformerBlock(self.D_MODEL, self.NUM_HEADS)
        self.bilstm      = BidirectionalLSTM(self.EMBED_DIM, self.HIDDEN)

        # حجم الدمج: bilstm_out + context
        combined_size = self.HIDDEN * 2 + self.CONTEXT_DIM
        self.classifier = Layer(combined_size, num_classes, activation=None)

        print(f"  📐 المعمارية:")
        print(f"     BPE vocab: {vocab_size}")
        print(f"     {self.embed}")
        print(f"     {self.transformer}")
        print(f"     {self.bilstm}")
        print(f"     Classifier: {combined_size} → {num_classes}")

    def _forward(self, ids, ctx_vec):
        """
        ids:     (batch, seq_len) — token IDs من BPE
        ctx_vec: (context_dim,)  — سياق المحادثة

        Pipeline:
            ids → Embedding(32) → TransformerBlock(32, heads=4)
                → BiLSTM(hidden=32) → concat(ctx) → Linear → Softmax
        """
        # 1. Embedding
        embedded = self.embed.forward(ids)  # (batch, seq, 32)

        # 2. Transformer (Self-Attention + FFN + Residual)
        trans_out = self.transformer.forward(embedded)  # (batch, seq, 32)

        # 3. BiLSTM — يقرأ خرج الـ Transformer من الاتجاهين
        bilstm_out = self.bilstm.forward(trans_out)  # (batch, hidden*2)

        # 4. دمج السياق
        batch = ids.shape[0]
        ctx   = np.tile(ctx_vec, (batch, 1))
        combined = np.concatenate([bilstm_out, ctx], axis=1)  # (batch, hidden*2+ctx)

        # 5. Classifier
        logits = self.classifier.forward(combined)  # (batch, num_classes)
        probs  = softmax(logits)
        return probs

    def train(self, epochs=200, verbose=True):
        print(f"\n⚙️  جاري التدريب ({epochs} epoch)...")
        num_classes = len(self.intents)
        X_ids = np.array([self.tokenizer.encode(x, self.SEQ_LEN) for x in self.X_data])
        Y_oh  = np.eye(num_classes)[self.y_data]

        history = []
        for epoch in range(1, epochs + 1):
            idx  = np.random.permutation(len(X_ids))
            X_sh = X_ids[idx]
            Y_sh = Y_oh[idx]
            total_loss = 0

            for i in range(len(X_sh)):
                x   = X_sh[i:i+1]
                y   = Y_sh[i:i+1]
                ctx = np.zeros(self.CONTEXT_DIM)

                # Forward
                probs = self._forward(x, ctx)
                eps   = 1e-9
                loss  = -np.sum(y * np.log(np.clip(probs, eps, 1-eps)))
                total_loss += loss

                # ── Backward الكامل عبر كل الشبكة ──
                grad = probs - y  # (1, num_classes)  [مشتقة Softmax+CrossEntropy]

                # 5 → 4: Classifier
                d_combined = self.classifier.backward(grad)
                if self.classifier.d_weights is not None:
                    self.optimizer.step(self.classifier.weights,
                                       self.classifier.d_weights)
                    self.optimizer.step(self.classifier.biases,
                                       self.classifier.d_biases)

                # فصل تدرج BiLSTM عن تدرج السياق (السياق ثابت أثناء التدريب)
                d_bilstm_out = d_combined[:, :self.HIDDEN * 2]

                # 4 → 3: BiLSTM
                d_trans_out = self.bilstm.backward(d_bilstm_out)
                self.bilstm.update(lr=0.03)

                # 3 → 2: Transformer
                d_embedded = self.transformer.backward(d_trans_out)
                self.transformer.update(lr=0.03)

                # 2 → 1: Embedding
                self.embed.backward(d_embedded)
                self.embed.update(lr=0.03)

            avg = total_loss / len(X_sh)
            history.append(avg)
            if verbose and (epoch % 50 == 0 or epoch == 1):
                bar = "█" * int(20 * (1 - avg/history[0])) if history[0] > 0 else ""
                print(f"  Epoch {epoch:3d} | Loss: {avg:.4f} | {bar}")

        print(f"✅ التدريب انتهى! Loss النهائي: {history[-1]:.4f}")
        self.trained = True
        return history

    def prepare_and_train(self, X_texts, y_labels, epochs=200):
        self.X_data = list(X_texts)
        self.y_data = list(y_labels)
        print("\n🔤 تدريب BPE Tokenizer...")
        self.tokenizer.fit(X_texts)
        print(f"  ✅ BPE: {self.tokenizer}")
        print("\n🏗️  بناء المعمارية...")
        self._build()
        return self.train(epochs)

    def chat(self, user_input):
        if not self.trained:
            return "النموذج لم يُدرَّب بعد.", 0, "unknown", 0.0

        ids = np.array([self.tokenizer.encode(user_input, self.SEQ_LEN)])
        ctx = self.memory.context_vector(self.CONTEXT_DIM)

        probs      = self._forward(ids, ctx)
        intent_idx = int(np.argmax(probs[0]))
        confidence = float(probs[0][intent_idx])
        intent_nm  = self.intents.get(intent_idx, "unknown")

        if confidence < 0.30:
            response = "لم أفهم جيداً، هل يمكنك إعادة الصياغة؟"
        else:
            response = random.choice(self.responses.get(intent_nm, ["..."]))

        self.memory.add(user_input, response, intent_idx, confidence)
        return response, intent_idx, intent_nm, confidence

    def learn(self, text, intent_idx):
        self.X_data.append(text)
        self.y_data.append(intent_idx)
        self.tokenizer.fit(self.X_data)
        self._build()
        self.train(epochs=150, verbose=False)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ✅ 7. SAVE / LOAD
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def save(self, path="zaatar_v3.pkl"):
        data = {
            "tokenizer_vocab":   self.tokenizer.vocab,
            "tokenizer_merges":  self.tokenizer.merges,
            "X_data":            self.X_data,
            "y_data":            self.y_data,
            "trained":           self.trained,
        }

        if self.embed is not None:
            data["embed_weights"] = self.embed.embeddings

        if self.classifier is not None:
            data["classifier_w"] = self.classifier.weights
            data["classifier_b"] = self.classifier.biases

        if self.transformer is not None:
            t = self.transformer
            data["transformer"] = {
                "attn_W_Q": t.attention.W_Q, "attn_W_K": t.attention.W_K,
                "attn_W_V": t.attention.W_V, "attn_W_O": t.attention.W_O,
                "norm1_gamma": t.norm1.gamma, "norm1_beta": t.norm1.beta,
                "norm2_gamma": t.norm2.gamma, "norm2_beta": t.norm2.beta,
                "ff_W1": t.ff.W1, "ff_b1": t.ff.b1,
                "ff_W2": t.ff.W2, "ff_b2": t.ff.b2,
            }

        if self.bilstm is not None:
            def lstm_weights(lstm):
                return {k: getattr(lstm, k) for k in
                        ['W_f','W_i','W_g','W_o','b_f','b_i','b_g','b_o']}
            data["bilstm"] = {
                "fwd": lstm_weights(self.bilstm.fwd),
                "bwd": lstm_weights(self.bilstm.bwd),
            }

        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 الأوزان محفوظة: {path}")

    def load(self, path="zaatar_v3.pkl"):
        if not os.path.exists(path):
            print(f"❌ الملف غير موجود: {path}")
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.tokenizer.vocab         = data["tokenizer_vocab"]
        self.tokenizer.merges        = data["tokenizer_merges"]
        self.tokenizer.inverse_vocab = {v: k for k, v in data["tokenizer_vocab"].items()}
        self.X_data  = data["X_data"]
        self.y_data  = data["y_data"]
        self.trained = data["trained"]

        self._build()

        if data.get("embed_weights") is not None:
            self.embed.embeddings = data["embed_weights"]

        if data.get("classifier_w") is not None:
            self.classifier.weights = data["classifier_w"]
            self.classifier.biases  = data["classifier_b"]

        if data.get("transformer") is not None:
            t  = self.transformer
            tw = data["transformer"]
            t.attention.W_Q = tw["attn_W_Q"]; t.attention.W_K = tw["attn_W_K"]
            t.attention.W_V = tw["attn_W_V"]; t.attention.W_O = tw["attn_W_O"]
            t.norm1.gamma   = tw["norm1_gamma"]; t.norm1.beta = tw["norm1_beta"]
            t.norm2.gamma   = tw["norm2_gamma"]; t.norm2.beta = tw["norm2_beta"]
            t.ff.W1 = tw["ff_W1"]; t.ff.b1 = tw["ff_b1"]
            t.ff.W2 = tw["ff_W2"]; t.ff.b2 = tw["ff_b2"]

        if data.get("bilstm") is not None:
            bw = data["bilstm"]
            for direction, lstm in [("fwd", self.bilstm.fwd), ("bwd", self.bilstm.bwd)]:
                for k, v in bw[direction].items():
                    setattr(lstm, k, v)

        print(f"✅ الأوزان محملة من: {path}")
        return True


# ══════════════════════════════════════════════════
# ✅ 6. FLASK REST API
# ══════════════════════════════════════════════════

def create_flask_app(bot):
    """
    REST API endpoints:
        POST /chat          → {"message":"مرحبا"} → رد زعتر
        POST /learn         → {"text":"...","intent":0}
        GET  /memory        → سياق المحادثة
        GET  /status        → حالة النموذج
        GET  /attention     → خريطة انتباه آخر رسالة
        DELETE /memory      → مسح الذاكرة
    """
    try:
        from flask import Flask, request, jsonify, render_template
    except ImportError:
        print("⚠️  Flask غير مثبت. شغّل: pip install flask flask-cors")
        return None

    app = Flask(__name__)

    try:
        from flask_cors import CORS
        CORS(app)  # السماح للواجهة بالتواصل مع الـ API
    except ImportError:
        print("⚠️  flask-cors غير مثبت — سيعمل API لكن بدون CORS.")
        print("   شغّل: pip install flask-cors  (أو استخدم نفس origin للواجهة)")
        # إضافة CORS يدوياً بدون المكتبة
        @app.after_request
        def add_cors_headers(response):
            response.headers["Access-Control-Allow-Origin"]  = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return response

    @app.route("/", methods=["GET"])
    def index():
        """يعرض واجهة الدردشة مباشرة من نفس السيرفر"""
        return render_template("zaatar_chat.html")

    @app.route("/status", methods=["GET"])
    def status():
        return jsonify({
            "status":     "running",
            "trained":    bot.trained,
            "vocab_size": bot.tokenizer.vocab_size,
            "bpe_merges": len(bot.tokenizer.merges),
            "memory_len": len(bot.memory.history),
            "architecture": {
                "tokenizer":   str(bot.tokenizer),
                "embedding":   f"EmbeddingLayer(vocab={bot.tokenizer.vocab_size}, dim={bot.EMBED_DIM})",
                "transformer": str(bot.transformer),
                "bilstm":      str(bot.bilstm),
                "classifier":  f"Linear({bot.HIDDEN*2+bot.CONTEXT_DIM} → {len(bot.intents)})"
            }
        })

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "أرسل {'message': 'نص الرسالة'}"}), 400

        msg    = data["message"].strip()
        if not msg:
            return jsonify({"error": "الرسالة فارغة"}), 400

        reply, intent_id, intent_name, conf = bot.chat(msg)

        return jsonify({
            "reply":       reply,
            "intent_id":   intent_id,
            "intent_name": intent_name,
            "confidence":  round(conf, 4),
            "memory_size": len(bot.memory.history),
            "context_used": True
        })

    @app.route("/learn", methods=["POST"])
    def learn():
        data = request.get_json()
        if not data or "text" not in data or "intent" not in data:
            return jsonify({"error": "أرسل {'text':'...','intent':0}"}), 400

        text       = data["text"].strip()
        intent_idx = int(data["intent"])

        if intent_idx not in bot.intents:
            return jsonify({"error": f"النية {intent_idx} غير موجودة"}), 400

        bot.learn(text, intent_idx)
        return jsonify({
            "status":      "تم التعلم",
            "text":        text,
            "intent":      bot.intents[intent_idx],
            "vocab_size":  bot.tokenizer.vocab_size
        })

    @app.route("/memory", methods=["GET"])
    def memory():
        return jsonify({
            "history": bot.memory.to_dict(),
            "size":    len(bot.memory.history)
        })

    @app.route("/memory", methods=["DELETE"])
    def clear_memory():
        bot.memory.history.clear()
        return jsonify({"status": "تم مسح الذاكرة"})

    @app.route("/attention", methods=["POST"])
    def attention():
        data = request.get_json()
        msg  = data.get("message", "")
        if not msg or not bot.trained:
            return jsonify({"error": "أرسل رسالة أولاً"}), 400

        ids = np.array([bot.tokenizer.encode(msg, bot.SEQ_LEN)])
        bot._forward(ids, np.zeros(bot.CONTEXT_DIM))

        attn = bot.transformer.get_attention_map()  # (1, heads, seq, seq)
        avg_attn = attn[0].mean(axis=0).tolist()    # متوسط الرؤوس

        return jsonify({
            "attention_matrix": avg_attn,
            "num_heads": bot.transformer.attention.num_heads,
            "message": msg
        })

    @app.route("/save", methods=["POST"])
    def save_model():
        bot.save("zaatar_v3.pkl")
        return jsonify({"status": "تم الحفظ", "file": "zaatar_v3.pkl"})

    return app


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("   AI_ENGINE Zaatar v3.0")
    print("   LSTM · BiRNN · Multi-Head Attention · Transformer · BPE")
    print("=" * 65)

    intents = {0: "greeting", 1: "goodbye", 2: "identity", 3: "thanks"}
    responses = {
        "greeting": ["أهلاً بك!", "مرحباً، كيف يمكنني مساعدتك؟", "يا هلا! 🌿"],
        "goodbye":  ["إلى اللقاء!", "مع السلامة يوماً رائعاً."],
        "identity": [
            "أنا زعتر v3 🧠 — بُنيت بـ LSTM + BiRNN + Transformer + BPE",
            "اسمي زعتر، محرك ذكاء اصطناعي مبني بالكامل من الصفر!"
        ],
        "thanks":   ["العفو!", "لا شكر على واجب.", "في خدمتك دائماً 🌿"]
    }

    X_train = [
        "مرحبا", "هلا", "السلام عليكم", "كيف الحال",
        "اهلا وسهلا", "صباح الخير", "مساء الخير", "هاي",
        "مع السلامة", "وداعا", "باي", "الى اللقاء",
        "يلا باي", "انتهى الكلام", "تصبح على خير",
        "من أنت", "ما اسمك", "عرف عن نفسك", "وش اسمك",
        "هل انت ذكاء اصطناعي", "كيف تعمل", "من صنعك",
        "شكرا", "يعطيك العافية", "تسلم", "الف شكر",
        "ممنون", "شكرا جزيلا", "مشكور"
    ]
    y_train = [
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3
    ]

    bot = ZaatarV3(intents, responses)

    # محاولة تحميل نموذج محفوظ
    if not bot.load("zaatar_v3.pkl"):
        bot.prepare_and_train(X_train, y_train, epochs=250)
        bot.save("zaatar_v3.pkl")

    # تشغيل Flask API
    app = create_flask_app(bot)
    port = int(os.environ.get("PORT", 7860))

    if app:
        print(f"\n🌐 الواجهة + API يعملان على: http://localhost:{port}")
        print("   افتح الرابط في المتصفح للمحادثة مباشرة 🌿")
        print("\n   Endpoints:")
        print("   GET  /          → واجهة الدردشة")
        print("   POST /chat      → {'message': '...'}")
        print("   GET  /status    → حالة النموذج")
        print("   GET  /memory    → ذاكرة المحادثة")
        print("   POST /learn     → {'text':'...','intent':0}")
        print("   POST /attention → خريطة الانتباه")
        print("   POST /save      → حفظ الأوزان")
        print("\n" + "─" * 65)

        # وضع المحادثة في Terminal — فقط إذا كان هناك طرفية تفاعلية فعلية
        # (يتجاوز هذا تلقائياً عند النشر على Render/Railway/Docker بدون stdin)
        if sys.stdin.isatty():
            print("💬 أو تحدث مباشرة هنا (Ctrl+C لتشغيل API فقط):")
            print("─" * 65)
            try:
                while True:
                    user = input("أنت: ").strip()
                    if not user:
                        continue
                    if user == "خروج":
                        bot.save("zaatar_v3.pkl")
                        break
                    if user == "حفظ":
                        bot.save("zaatar_v3.pkl")
                        continue

                    reply, iid, inm, conf = bot.chat(user)
                    print(f"زعتر: {reply}")
                    print(f"       [النية:{inm} | ثقة:{conf:.1%} | ذاكرة:{len(bot.memory.history)}]")
            except KeyboardInterrupt:
                print("\n\n🌐 تشغيل API فقط...")

        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        # CLI فقط إذا لم يكن Flask متاحاً
        print("\n💬 وضع المحادثة (اكتب 'خروج' للإنهاء):")
        while True:
            user = input("أنت: ").strip()
            if user == "خروج":
                bot.save()
                break
            reply, iid, inm, conf = bot.chat(user)
            print(f"زعتر: {reply}  [{inm} | {conf:.1%}]")


if __name__ == "__main__":
    main()
