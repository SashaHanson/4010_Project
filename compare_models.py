import modal

app = modal.App("model-compare")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "matplotlib", "scikit-learn")
)

volume = modal.Volume.from_name("dataset")

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 30,
    volumes={"/data": volume},
)
def compare_models():
    import torch
    import torch.nn as nn
    import numpy as np
    import torch.fft as fft
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import os
    import math

    # Load
    X = np.load("/data/X.npy")
    Y = np.load("/data/Y.npy")

    input_dim = X.shape[2]      # 16
    output_dim = Y.shape[2]     # 6
    pred_len = Y.shape[1]       # 168

    X_t = torch.tensor(X, dtype=torch.float32).cuda()

    # -------------------------
    # LSTM MODEL
    # -------------------------
    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden=128, layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True)
        def forward(self, x):
            _, (h, c) = self.lstm(x)
            return h, c

    class Decoder(nn.Module):
        def __init__(self, output_dim, hidden=128, layers=2, seq_len=168):
            super().__init__()
            self.seq_len = seq_len
            self.lstm = nn.LSTM(output_dim, hidden, layers, batch_first=True)
            self.fc = nn.Linear(hidden, output_dim)
        def forward(self, h, c):
            B = h.shape[1]
            zeros = torch.zeros((B, self.seq_len, output_dim), device=h.device)
            out,_ = self.lstm(zeros, (h,c))
            return self.fc(out)

    class Seq2Seq(nn.Module):
        def __init__(self, input_dim, output_dim, pred_len):
            super().__init__()
            self.encoder = Encoder(input_dim)
            self.decoder = Decoder(output_dim, seq_len=pred_len)
        def forward(self, x):
            h,c = self.encoder(x)
            return self.decoder(h,c)

    lstm = Seq2Seq(input_dim, output_dim, pred_len).cuda()
    lstm.load_state_dict(torch.load("/data/lstm_weather_model.pth"))
    lstm.eval()

    # -------------------------
    # FEDFORMER (Proper Implementation)
    # -------------------------
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]

    class FourierBlock(nn.Module):
        def __init__(self, d_model, modes=32, dropout=0.1):
            super().__init__()
            self.modes = modes
            scale = 1 / (d_model * modes)
            self.weights_real = nn.Parameter(scale * torch.randn(modes, d_model))
            self.weights_imag = nn.Parameter(scale * torch.randn(modes, d_model))
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(d_model)
        def forward(self, x):
            B, L, C = x.shape
            residual = x
            xf = fft.rfft(x, dim=1)
            kept = xf[:, :self.modes, :]
            real = kept.real * self.weights_real - kept.imag * self.weights_imag
            imag = kept.real * self.weights_imag + kept.imag * self.weights_real
            mixed = real + 1j * imag
            out = torch.zeros_like(xf)
            out[:, :self.modes, :] = mixed
            x = fft.irfft(out, n=L, dim=1)
            x = self.norm(x + residual)
            x = self.dropout(x)
            return x

    class FEDformerEncoder(nn.Module):
        def __init__(self, d_model, n_layers=3, modes=32, dropout=0.1):
            super().__init__()
            self.layers = nn.ModuleList([
                FourierBlock(d_model, modes, dropout) for _ in range(n_layers)
            ])
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class FEDformerDecoder(nn.Module):
        def __init__(self, d_model, pred_len, modes=32, dropout=0.1):
            super().__init__()
            self.pred_len = pred_len
            self.modes = modes
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(d_model)
            self.cross_attention = nn.MultiheadAttention(
                d_model, num_heads=8, dropout=dropout, batch_first=True
            )
            self.fourier_block = FourierBlock(d_model, modes, dropout)
        def forward(self, decoder_input, encoder_output):
            attn_out, _ = self.cross_attention(
                decoder_input, encoder_output, encoder_output
            )
            decoder_input = decoder_input + attn_out
            x = self.fourier_block(decoder_input)
            return x

    class FEDformer(nn.Module):
        def __init__(self, input_dim, output_dim, pred_len, 
                     d_model=256, n_encoder_layers=3, n_decoder_layers=2, 
                     modes=32, dropout=0.1):
            super().__init__()
            self.pred_len = pred_len
            self.d_model = d_model
            self.embed = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model)
            self.encoder = FEDformerEncoder(d_model, n_encoder_layers, modes, dropout)
            self.decoder_layers = nn.ModuleList([
                FEDformerDecoder(d_model, pred_len, modes, dropout) 
                for _ in range(n_decoder_layers)
            ])
            self.output_projection = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, output_dim)
            )
        def forward(self, x):
            batch_size = x.shape[0]
            x = self.embed(x)
            x = self.pos_encoder(x)
            encoder_output = self.encoder(x)
            last_hidden = encoder_output[:, -1:, :]
            decoder_pos = self.pos_encoder.pe[:, :self.pred_len, :]
            decoder_input = last_hidden.repeat(1, self.pred_len, 1) + decoder_pos
            for decoder_layer in self.decoder_layers:
                decoder_input = decoder_layer(decoder_input, encoder_output)
            output = self.output_projection(decoder_input)
            return output

    fed = FEDformer(
        input_dim=input_dim,
        output_dim=output_dim,
        pred_len=pred_len,
        d_model=256,
        n_encoder_layers=3,
        n_decoder_layers=2,
        modes=32,
        dropout=0.1
    ).cuda()
    fed.load_state_dict(torch.load("/data/fedformer_weather_model.pth"))
    fed.eval()

    # -------------------------
    # EVALUATE BOTH MODELS
    # -------------------------
    bs = 512
    lstm_preds = []
    fed_preds = []

    with torch.no_grad():
        for i in range(0, len(X_t), bs):
            lstm_preds.append(lstm(X_t[i:i+bs]).cpu().numpy())
            fed_preds.append(fed(X_t[i:i+bs]).cpu().numpy())

    lstm_preds = np.concatenate(lstm_preds, axis=0)
    fed_preds  = np.concatenate(fed_preds,  axis=0)

    true_last = Y[:, -1]
    lstm_last = lstm_preds[:, -1]
    fed_last  = fed_preds[:, -1]

    print("\n" + "=" * 60)
    print("MODEL COMPARISON: LSTM vs FEDformer")
    print("=" * 60)
    print(f"{'Metric':<12} {'LSTM':<15} {'FEDformer':<15} {'Winner':<10}")
    print("-" * 60)
    
    metrics = [
        ("MAE", mean_absolute_error(true_last, lstm_last), mean_absolute_error(true_last, fed_last), "lower"),
        ("MSE", mean_squared_error(true_last, lstm_last), mean_squared_error(true_last, fed_last), "lower"),
        ("RMSE", np.sqrt(mean_squared_error(true_last, lstm_last)), np.sqrt(mean_squared_error(true_last, fed_last)), "lower"),
        ("R²", r2_score(true_last, lstm_last), r2_score(true_last, fed_last), "higher"),
    ]
    
    for name, lstm_val, fed_val, better in metrics:
        if better == "lower":
            winner = "FEDformer" if fed_val < lstm_val else "LSTM"
            lstm_str = f"{lstm_val:.6f}"
            fed_str = f"{fed_val:.6f}"
        else:
            winner = "FEDformer" if fed_val > lstm_val else "LSTM"
            lstm_str = f"{lstm_val:.6f}"
            fed_str = f"{fed_val:.6f}"
        
        print(f"{name:<12} {lstm_str:<15} {fed_str:<15} {winner:<10}")

    # Calculate improvement percentages
    mae_improvement = ((mean_absolute_error(true_last, lstm_last) - mean_absolute_error(true_last, fed_last)) / mean_absolute_error(true_last, lstm_last)) * 100
    rmse_improvement = ((np.sqrt(mean_squared_error(true_last, lstm_last)) - np.sqrt(mean_squared_error(true_last, fed_last))) / np.sqrt(mean_squared_error(true_last, lstm_last))) * 100
    r2_improvement = ((r2_score(true_last, fed_last) - r2_score(true_last, lstm_last)) / abs(r2_score(true_last, lstm_last))) * 100 if r2_score(true_last, lstm_last) != 0 else 0

    print("\n" + "=" * 60)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 60)
    print(f"MAE Improvement:  {mae_improvement:+.2f}% ({'FEDformer better' if mae_improvement > 0 else 'LSTM better'})")
    print(f"RMSE Improvement: {rmse_improvement:+.2f}% ({'FEDformer better' if rmse_improvement > 0 else 'LSTM better'})")
    print(f"R² Improvement:   {r2_improvement:+.2f}% ({'FEDformer better' if r2_improvement > 0 else 'LSTM better'})")

@app.local_entrypoint()
def main():
    print("Starting model comparison...")
    compare_models.remote()
    print("Comparison completed!")

