import modal

app = modal.App("fedformer-eval")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "scikit-learn", "matplotlib")
)

volume = modal.Volume.from_name("dataset")

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 30,
    volumes={"/data": volume},
)
def test_transformer():
    import torch
    import torch.nn as nn
    import numpy as np
    import torch.fft as fft
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import os
    import math

    X = np.load("/data/X.npy")
    Y = np.load("/data/Y.npy")

    input_dim = X.shape[2]      # 16
    output_dim = Y.shape[2]     # 6
    pred_len = Y.shape[1]       # 168

    X_t = torch.tensor(X, dtype=torch.float32).cuda()

    # Same model as training - FEDformer with proper architecture
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

    model = FEDformer(
        input_dim=input_dim,
        output_dim=output_dim,
        pred_len=pred_len,
        d_model=256,
        n_encoder_layers=3,
        n_decoder_layers=2,
        modes=32,
        dropout=0.1
    ).cuda()
    model.load_state_dict(torch.load("/data/fedformer_weather_model.pth"))
    model.eval()

    preds = []
    bs = 512

    with torch.no_grad():
        for i in range(0, len(X_t), bs):
            preds.append(model(X_t[i:i+bs]).cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    true_last = Y[:, -1]
    pred_last = preds[:, -1]

    print("\n===== FEDFORMER MODEL PERFORMANCE =====")
    print("MAE :", mean_absolute_error(true_last, pred_last))
    print("MSE :", mean_squared_error(true_last, pred_last))
    print("RMSE:", np.sqrt(mean_squared_error(true_last, pred_last)))
    print("R2  :", r2_score(true_last, pred_last))

@app.local_entrypoint()
def main():
    print("Starting FEDformer evaluation...")
    test_transformer.remote()
    print("Evaluation completed!")

