"""
src/models.py
─────────────
All 7 RNN model architectures.

Each builder function returns a compiled keras.Model.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


# ─────────────────────────────────────────────────────────────────────────────
# 1  Vanilla RNN
# ─────────────────────────────────────────────────────────────────────────────
def build_vanilla_rnn(seq_len: int, units: int = 64, dropout: float = 0.2) -> keras.Model:
    """
    Baseline model.  Uses tanh SimpleRNN — subject to vanishing gradients
    on sequences longer than ~10-15 steps.
    """
    model = keras.Sequential([
        layers.SimpleRNN(units, activation="tanh",
                         input_shape=(seq_len, 1),
                         name="simple_rnn"),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ], name="Vanilla_RNN")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="mse", metrics=["mae"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2  LSTM
# ─────────────────────────────────────────────────────────────────────────────
def build_lstm(seq_len: int, units: int = 64, dropout: float = 0.2) -> keras.Model:
    """
    Single-layer LSTM.  The forget / input / output gates give the cell state
    a near-constant gradient path — solving vanishing gradients.
    """
    model = keras.Sequential([
        layers.LSTM(units, input_shape=(seq_len, 1), name="lstm"),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ], name="LSTM")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="mse", metrics=["mae"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3  GRU
# ─────────────────────────────────────────────────────────────────────────────
def build_gru(seq_len: int, units: int = 64, dropout: float = 0.2) -> keras.Model:
    """
    GRU merges the forget + input gates into a single 'update gate' and adds
    a 'reset gate'.  ~25 % fewer parameters than LSTM, often comparable quality.
    """
    model = keras.Sequential([
        layers.GRU(units, input_shape=(seq_len, 1), name="gru"),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ], name="GRU")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="mse", metrics=["mae"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4  Stacked LSTM
# ─────────────────────────────────────────────────────────────────────────────
def build_stacked_lstm(seq_len: int, units: int = 64,
                       dropout: float = 0.2) -> keras.Model:
    """
    3-layer deep LSTM.  Each layer learns temporal patterns at a different
    resolution.  Intermediate layers must use return_sequences=True so that
    the next layer receives a full sequence, not just the final hidden state.
    """
    model = keras.Sequential([
        layers.LSTM(units,       return_sequences=True,
                    input_shape=(seq_len, 1), name="lstm_1"),
        layers.Dropout(dropout),
        layers.LSTM(units // 2,  return_sequences=True, name="lstm_2"),
        layers.Dropout(dropout),
        layers.LSTM(units // 4,  name="lstm_3"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ], name="Stacked_LSTM")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="mse", metrics=["mae"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5  Bidirectional LSTM
# ─────────────────────────────────────────────────────────────────────────────
def build_bilstm(seq_len: int, units: int = 64, dropout: float = 0.2) -> keras.Model:
    """
    Runs two LSTMs — one left-to-right, one right-to-left — and concatenates
    their hidden states.  Useful when the full context around each time step
    is available (classification, anomaly detection).
    """
    model = keras.Sequential([
        layers.Bidirectional(layers.LSTM(units), input_shape=(seq_len, 1),
                             name="bilstm"),
        layers.Dropout(dropout),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ], name="Bidirectional_LSTM")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="mse", metrics=["mae"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6  Attention-LSTM  (Bahdanau additive attention)
# ─────────────────────────────────────────────────────────────────────────────
class BahdanauAttention(layers.Layer):
    """
    Additive (Bahdanau) attention.

    Score:    e_t  = V · tanh(W · h_t)
    Weights:  α_t  = softmax(e_t)            over the time dimension
    Context:  c    = Σ_t  α_t · h_t
    """
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1,     use_bias=False)

    def call(self, encoder_outputs):           # (B, T, H)
        score   = self.V(tf.nn.tanh(self.W(encoder_outputs)))   # (B, T, 1)
        alpha   = tf.nn.softmax(score, axis=1)                   # (B, T, 1)
        context = tf.reduce_sum(alpha * encoder_outputs, axis=1) # (B, H)
        return context, alpha                                     # also return weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.W.units})
        return cfg


def build_attention_lstm(seq_len: int, lstm_units: int = 64,
                         attn_units: int = 32,
                         dropout: float = 0.2) -> keras.Model:
    """
    LSTM that returns all hidden states, then applies Bahdanau attention so
    the model can selectively weight each time step before prediction.
    """
    inp      = layers.Input(shape=(seq_len, 1), name="input")
    lstm_out = layers.LSTM(lstm_units, return_sequences=True,
                           name="encoder_lstm")(inp)
    lstm_out = layers.Dropout(dropout)(lstm_out)
    ctx, _   = BahdanauAttention(attn_units, name="attention")(lstm_out)
    out      = layers.Dense(64, activation="relu")(ctx)
    out      = layers.Dense(1,  name="output")(out)

    model = Model(inp, out, name="Attention_LSTM")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="mse", metrics=["mae"])
    return model


# Helper: extract attention weights for a batch of inputs
def get_attention_weights(attention_model: keras.Model,
                          X_sample: "np.ndarray") -> "np.ndarray":
    """Return α weights of shape (N, seq_len) from a trained Attention-LSTM."""
    inp       = attention_model.input
    lstm_out  = attention_model.get_layer("encoder_lstm").output
    attn_layer= attention_model.get_layer("attention")
    _, alpha  = attn_layer(lstm_out)
    debug_model = Model(inputs=inp, outputs=alpha)
    weights = debug_model.predict(X_sample, verbose=0)  # (N, T, 1)
    return weights.squeeze(-1)                           # (N, T)


# ─────────────────────────────────────────────────────────────────────────────
# 7  Encoder-Decoder (Seq2Seq)
# ─────────────────────────────────────────────────────────────────────────────
def build_seq2seq(seq_len: int, pred_len: int,
                  units: int = 64, dropout: float = 0.2) -> keras.Model:
    """
    Classic LSTM Encoder-Decoder for multi-step forecasting.

    Architecture
    ────────────
    Encoder  : LSTM reads the 30-day input → produces context (h_T, c_T)
    Bridge   : RepeatVector repeats the context vector pred_len times
    Decoder  : LSTM generates pred_len hidden states  →  TimeDistributed Dense

    Output shape: (batch, pred_len, 1)  →  squeezed to (batch, pred_len)
    """
    model = keras.Sequential([
        # Encoder
        layers.LSTM(units, input_shape=(seq_len, 1),
                    return_sequences=False, name="encoder"),
        layers.Dropout(dropout),

        # Bridge
        layers.RepeatVector(pred_len, name="repeat"),  # (B, pred_len, units)

        # Decoder
        layers.LSTM(units, return_sequences=True, name="decoder"),
        layers.Dropout(dropout),

        # Output projection — one Dense per time step
        layers.TimeDistributed(layers.Dense(1), name="td_dense"),
    ], name="Seq2Seq_EncoderDecoder")

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="mse", metrics=["mae"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
def get_model(name: str, seq_len: int, pred_len: int = 7,
              units: int = 64, dropout: float = 0.2) -> keras.Model:
    """Factory function — returns a freshly compiled model by name."""
    dispatch = {
        "Vanilla RNN"        : lambda: build_vanilla_rnn(seq_len, units, dropout),
        "LSTM"               : lambda: build_lstm(seq_len, units, dropout),
        "GRU"                : lambda: build_gru(seq_len, units, dropout),
        "Stacked LSTM"       : lambda: build_stacked_lstm(seq_len, units, dropout),
        "Bidirectional LSTM" : lambda: build_bilstm(seq_len, units, dropout),
        "Attention LSTM"     : lambda: build_attention_lstm(seq_len, units,
                                                             units // 2, dropout),
        "Seq2Seq"            : lambda: build_seq2seq(seq_len, pred_len,
                                                      units, dropout),
    }
    if name not in dispatch:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(dispatch)}")
    return dispatch[name]()
