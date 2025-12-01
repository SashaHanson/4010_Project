import modal

app = modal.App("fedformer-training")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
)

volume = modal.Volume.from_name("dataset")


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 120,
    volumes={"/data": volume},
)
def train_transformer():
    import torch
    import torch.nn as nn
    import numpy as np
    import torch.fft as fft
    import os
    import math

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Files in /data:", os.listdir("/data"))

    # Load dataset
    X = np.load("/data/X.npy")
    Y = np.load("/data/Y.npy")

    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, Y),
        batch_size=32,
        shuffle=True
    )

    input_dim = X.shape[2]      # 16 features
    output_dim = Y.shape[2]     # 6 target features
    pred_len = Y.shape[1]       # 168 timesteps

    # --------------------------------------------------------------
    # FEDFORMER (Proper Implementation)
    # --------------------------------------------------------------

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
            
            # FFT
            xf = fft.rfft(x, dim=1)
            kept = xf[:, :self.modes, :]

            # Complex multiplication in frequency domain
            real = kept.real * self.weights_real - kept.imag * self.weights_imag
            imag = kept.real * self.weights_imag + kept.imag * self.weights_real
            mixed = real + 1j * imag

            # IFFT
            out = torch.zeros_like(xf)
            out[:, :self.modes, :] = mixed
            x = fft.irfft(out, n=L, dim=1)
            
            # Residual connection and normalization
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
            
            # Cross-attention for encoder-decoder interaction
            self.cross_attention = nn.MultiheadAttention(
                d_model, num_heads=8, dropout=dropout, batch_first=True
            )
            
            # Fourier block for decoder
            self.fourier_block = FourierBlock(d_model, modes, dropout)

        def forward(self, decoder_input, encoder_output):
            # Cross-attention: decoder queries attend to encoder keys/values
            attn_out, _ = self.cross_attention(
                decoder_input, encoder_output, encoder_output
            )
            decoder_input = decoder_input + attn_out
            
            # Apply Fourier block
            x = self.fourier_block(decoder_input)
            return x

    class FEDformer(nn.Module):
        def __init__(self, input_dim, output_dim, pred_len, 
                     d_model=256, n_encoder_layers=3, n_decoder_layers=2, 
                     modes=32, dropout=0.1):
            super().__init__()
            self.pred_len = pred_len
            self.d_model = d_model

            # Input embedding
            self.embed = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model)
            
            # Encoder
            self.encoder = FEDformerEncoder(d_model, n_encoder_layers, modes, dropout)
            
            # Decoder
            self.decoder_layers = nn.ModuleList([
                FEDformerDecoder(d_model, pred_len, modes, dropout) 
                for _ in range(n_decoder_layers)
            ])
            
            # Output projection
            self.output_projection = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, output_dim)
            )

        def forward(self, x):
            # x: [batch, seq_len, input_dim]
            batch_size = x.shape[0]
            
            # Encode input sequence
            x = self.embed(x)  # [batch, seq_len, d_model]
            x = self.pos_encoder(x)
            encoder_output = self.encoder(x)  # [batch, seq_len, d_model]
            
            # Create decoder input: use last timestep + positional encoding for future
            last_hidden = encoder_output[:, -1:, :]  # [batch, 1, d_model]
            
            # Create decoder queries with positional encoding for future timesteps
            decoder_pos = self.pos_encoder.pe[:, :self.pred_len, :]
            decoder_input = last_hidden.repeat(1, self.pred_len, 1) + decoder_pos
            
            # Decode to generate future sequence
            for decoder_layer in self.decoder_layers:
                decoder_input = decoder_layer(decoder_input, encoder_output)
            
            # Project to output dimension
            output = self.output_projection(decoder_input)  # [batch, pred_len, output_dim]
            
            return output

    # Build model with proper architecture
    model = FEDformer(
        input_dim=input_dim,
        output_dim=output_dim,
        pred_len=pred_len,
        d_model=256,
        n_encoder_layers=3,
        n_decoder_layers=2,
        modes=32,
        dropout=0.1
    ).to(device)

    # Better optimizer settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    criterion = nn.MSELoss()

    # Training loop with early stopping
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(30):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/30  Loss: {avg_loss:.6f}  LR: {current_lr:.2e}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    torch.save(model.state_dict(), "/data/fedformer_weather_model.pth")
    print(f"Saved /data/fedformer_weather_model.pth")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

@app.local_entrypoint()
def main():
    print("Starting FEDformer training...")
    train_transformer.remote()
    print("Training completed!")