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
    import matplotlib.pyplot as plt

    print("Files available in /data:", os.listdir("/data"))

    # Make plot directory for FEDformer
    plot_dir = "/data/plots_fedformer"
    os.makedirs(plot_dir, exist_ok=True)

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

    true_last = Y[:, -1, :]
    pred_last = preds[:, -1, :]

    mae = mean_absolute_error(true_last, pred_last)
    mse = mean_squared_error(true_last, pred_last)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_last, pred_last)

    print("\n===== FEDFORMER MODEL PERFORMANCE =====")
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)

    # --------------------------------------------
    # CREATE COMBINED PLOT - ALL GRAPHS IN ONE PNG
    # --------------------------------------------

    features = [f"Feature {i}" for i in range(output_dim)]
    errors = pred_last - true_last
    
    # Calculate R² per feature
    r2_features = []
    for f in range(output_dim):
        r2_f = r2_score(true_last[:, f], pred_last[:, f])
        r2_features.append(r2_f)
    
    # Calculate MAE across forecast horizon
    horizon_mae = []
    for t in range(pred_len):
        horizon_mae.append(mean_absolute_error(Y[:, t, :], preds[:, t, :]))

    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Forecast over time (168 hours) - Sample prediction vs actual
    ax1 = fig.add_subplot(gs[0, :])
    # Select a random sample to visualize
    sample_idx = len(preds) // 2  # Use middle sample
    hours = np.arange(1, pred_len + 1)
    
    # Plot first 3 features for visibility
    for f in range(min(3, output_dim)):
        ax1.plot(hours, Y[sample_idx, :, f], label=f'True Feature {f}', linewidth=2, alpha=0.7)
        ax1.plot(hours, preds[sample_idx, :, f], label=f'Predicted Feature {f}', linewidth=2, linestyle='--', alpha=0.7)
    ax1.set_xlabel('Forecast Hour', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title(f'FEDformer: Forecast Over 168 Hours (Sample {sample_idx})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Scatter plot - Predicted vs True (last timestep) - Combined for all features
    ax2 = fig.add_subplot(gs[1, 0])
    colors = plt.cm.tab10(np.linspace(0, 1, output_dim))
    for f in range(output_dim):
        ax2.scatter(true_last[:, f], pred_last[:, f], s=1, alpha=0.3, label=f'F{f}', c=[colors[f]])
    # Add diagonal line
    min_val = min(true_last.min(), pred_last.min())
    max_val = max(true_last.max(), pred_last.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('True Values', fontsize=10)
    ax2.set_ylabel('Predicted Values', fontsize=10)
    ax2.set_title('Predicted vs True (Last Step)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # 3. Error histogram
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(errors.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax3.set_xlabel('Error', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Error Distribution (Last Step)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. R² per feature
    ax4 = fig.add_subplot(gs[1, 2])
    bars = ax4.bar(features, r2_features, color=colors[:output_dim], edgecolor='black')
    ax4.set_ylabel('R² Score', fontsize=10)
    ax4.set_title('R² Per Feature (Last Step)', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 1])
    ax4.tick_params(axis='x', rotation=45)
    # Add value labels on bars
    for bar, val in zip(bars, r2_features):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. MAE across forecast horizon
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(horizon_mae, linewidth=2, color='darkblue')
    ax5.set_xlabel('Forecast Step (Hours)', fontsize=10)
    ax5.set_ylabel('MAE', fontsize=10)
    ax5.set_title('MAE over Forecast Horizon (1 → 168 hours)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Metrics summary text
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    metrics_text = f"""
    FEDFORMER MODEL PERFORMANCE SUMMARY
    
    Overall Metrics (Last Timestep):
    • MAE:  {mae:.6f}
    • MSE:  {mse:.6f}
    • RMSE: {rmse:.6f}
    • R²:   {r2:.6f}
    
    Best Feature R²: {max(r2_features):.4f} (Feature {np.argmax(r2_features)})
    Worst Feature R²: {min(r2_features):.4f} (Feature {np.argmin(r2_features)})
    
    Forecast Horizon:
    • Initial MAE (Hour 1): {horizon_mae[0]:.6f}
    • Final MAE (Hour 168): {horizon_mae[-1]:.6f}
    • Avg MAE: {np.mean(horizon_mae):.6f}
    """
    ax6.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 7. Feature-wise MAE comparison
    ax7 = fig.add_subplot(gs[2, 2])
    feature_mae = [mean_absolute_error(true_last[:, f], pred_last[:, f]) for f in range(output_dim)]
    bars = ax7.bar(features, feature_mae, color=colors[:output_dim], edgecolor='black')
    ax7.set_ylabel('MAE', fontsize=10)
    ax7.set_title('MAE Per Feature (Last Step)', fontsize=12, fontweight='bold')
    ax7.tick_params(axis='x', rotation=45)
    # Add value labels
    for bar, val in zip(bars, feature_mae):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax7.grid(True, alpha=0.3, axis='y')

    # Add overall title
    fig.suptitle('FEDformer Model Evaluation - Complete Analysis', fontsize=16, fontweight='bold', y=0.995)

    # Save combined plot
    plt.savefig(f"{plot_dir}/fedformer_complete_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\nSaved combined plot file:", f"{plot_dir}/fedformer_complete_analysis.png")
    print("FEDformer evaluation and graphing completed.")

@app.local_entrypoint()
def main():
    print("Starting FEDformer evaluation...")
    test_transformer.remote()
    print("Evaluation completed!")

