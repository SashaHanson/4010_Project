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
    seq_len = X.shape[1]        # 240

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
    # TCNN MODEL
    # -------------------------
    class TemporalConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.pad1 = nn.ConstantPad1d((padding, 0), 0)
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            self.pad2 = nn.ConstantPad1d((padding, 0), 0)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        def forward(self, x):
            out = self.pad1(x)
            out = self.conv1(out)
            out = self.relu1(out)
            out = self.dropout1(out)
            out = self.pad2(out)
            out = self.conv2(out)
            out = self.relu2(out)
            out = self.dropout2(out)
            res = x if self.residual is None else self.residual(x)
            return out + res

    class TCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Match the saved model architecture: [16, 128, 128, 128, ...] with 7 layers
            channels = [input_dim, 128, 128, 128, 128, 128, 128, 128]
            layers = []
            for i in range(len(channels) - 1):
                layers.append(TemporalConvBlock(channels[i], channels[i + 1], kernel_size=3, dilation=2 ** i, dropout=0.1))
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Linear(channels[-1], pred_len * output_dim)
        def forward(self, x):
            x = x.transpose(1, 2)
            features = self.tcn(x)
            last_step = features[:, :, -1]
            out = self.head(last_step)
            return out.view(-1, pred_len, output_dim)

    tcnn = TCNN().cuda()
    tcnn.load_state_dict(torch.load("/data/tcnn_weather_model.pth"))
    tcnn.eval()

    # -------------------------
    # IMPROVED FEDFORMER ARCHITECTURE
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
        def forward(self, x, start_idx=0):
            return x + self.pe[:, start_idx:start_idx + x.size(1), :]

    class FourierBlock(nn.Module):
        def __init__(self, d_model, modes=64, dropout=0.1):
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
        def __init__(self, d_model, n_layers=4, modes=64, dropout=0.1):
            super().__init__()
            self.layers = nn.ModuleList([
                FourierBlock(d_model, modes, dropout) for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.norm(x)

    class FEDformerDecoder(nn.Module):
        def __init__(self, d_model, pred_len, modes=64, dropout=0.1):
            super().__init__()
            self.pred_len = pred_len
            self.modes = modes
            self.dropout = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            
            # Self-attention in decoder
            self.self_attention = nn.MultiheadAttention(
                d_model, num_heads=16, dropout=dropout, batch_first=True
            )
            
            # Cross-attention for encoder-decoder interaction
            self.cross_attention = nn.MultiheadAttention(
                d_model, num_heads=16, dropout=dropout, batch_first=True
            )
            
            # Feed-forward network
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model)
            )
            
            # Fourier block for decoder
            self.fourier_block = FourierBlock(d_model, modes, dropout)
        def forward(self, decoder_input, encoder_output):
            # Self-attention
            residual = decoder_input
            self_attn_out, _ = self.self_attention(
                decoder_input, decoder_input, decoder_input
            )
            decoder_input = self.norm1(decoder_input + self_attn_out)
            
            # Cross-attention: decoder queries attend to encoder keys/values
            residual = decoder_input
            cross_attn_out, _ = self.cross_attention(
                decoder_input, encoder_output, encoder_output
            )
            decoder_input = self.norm2(decoder_input + cross_attn_out)
            
            # Feed-forward
            residual = decoder_input
            ff_out = self.ff(decoder_input)
            decoder_input = self.norm3(decoder_input + ff_out)
            
            # Apply Fourier block
            x = self.fourier_block(decoder_input)
            return x

    class FEDformer(nn.Module):
        def __init__(self, input_dim, output_dim, pred_len, seq_len,
                     d_model=512, n_encoder_layers=4, n_decoder_layers=3, 
                     modes=64, dropout=0.1):
            super().__init__()
            self.pred_len = pred_len
            self.d_model = d_model

            # Input embedding
            self.embed = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + pred_len)
            
            # Encoder
            self.encoder = FEDformerEncoder(d_model, n_encoder_layers, modes, dropout)
            
            # Learned decoder query initialization
            self.decoder_query_proj = nn.Linear(d_model, d_model)
            
            # Decoder
            self.decoder_layers = nn.ModuleList([
                FEDformerDecoder(d_model, pred_len, modes, dropout) 
                for _ in range(n_decoder_layers)
            ])
            
            # Improved output projection: 3-layer MLP with residual
            self.output_proj1 = nn.Linear(d_model, d_model * 2)
            self.output_proj2 = nn.Linear(d_model * 2, d_model)
            self.output_proj3 = nn.Linear(d_model, output_dim)
            self.output_norm = nn.LayerNorm(d_model)
            self.output_activation = nn.GELU()
            self.output_dropout = nn.Dropout(dropout)

        def forward(self, x):
            # x: [batch, seq_len, input_dim]
            batch_size = x.shape[0]
            
            # Encode input sequence
            x = self.embed(x)  # [batch, seq_len, d_model]
            x = self.pos_encoder(x)
            encoder_output = self.encoder(x)  # [batch, seq_len, d_model]
            
            # Improved decoder initialization: learned projection from encoder output
            last_hidden = encoder_output[:, -1:, :]  # [batch, 1, d_model]
            decoder_base = self.decoder_query_proj(last_hidden)  # [batch, 1, d_model]
            
            # Create decoder queries with positional encoding for future timesteps
            decoder_pos = self.pos_encoder.pe[:, seq_len:seq_len + self.pred_len, :]
            decoder_input = decoder_base.repeat(1, self.pred_len, 1) + decoder_pos
            
            # Decode to generate future sequence
            for decoder_layer in self.decoder_layers:
                decoder_input = decoder_layer(decoder_input, encoder_output)
            
            # Improved output projection with residual connection
            residual = decoder_input
            out = self.output_proj1(decoder_input)
            out = self.output_activation(out)
            out = self.output_dropout(out)
            out = self.output_proj2(out)
            out = self.output_norm(out + residual)  # Residual connection
            out = self.output_proj3(out)
            
            return out

    # Try to load checkpoint and detect architecture
    checkpoint = torch.load("/data/fedformer_weather_model.pth", map_location='cuda')
    
    # Check if it's a new checkpoint with config
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        config = checkpoint['model_config']
        fed = FEDformer(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            pred_len=config['pred_len'],
            seq_len=config['seq_len'],
            d_model=config['d_model'],
            n_encoder_layers=config['n_encoder_layers'],
            n_decoder_layers=config['n_decoder_layers'],
            modes=config['modes'],
            dropout=config['dropout']
        ).cuda()
        fed.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded improved FEDformer (d_model={config['d_model']}, encoder_layers={config['n_encoder_layers']}, decoder_layers={config['n_decoder_layers']})")
    else:
        # Try to detect architecture from checkpoint
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        
        # Detect architecture by checking embed weight shape and key presence
        d_model = state_dict.get('embed.weight', torch.zeros(1, 1)).shape[0]
        has_new_keys = any('decoder_query_proj' in k or 'output_proj1' in k for k in state_dict.keys())
        is_new_arch = (d_model == 512) and has_new_keys
        
        if is_new_arch:
            # New improved architecture
            fed = FEDformer(
                input_dim=input_dim,
                output_dim=output_dim,
                pred_len=pred_len,
                seq_len=seq_len,
                d_model=512,
                n_encoder_layers=4,
                n_decoder_layers=3,
                modes=64,
                dropout=0.1
            ).cuda()
            fed.load_state_dict(state_dict)
            print("Loaded improved FEDformer architecture")
        else:
            # Old architecture - need to use old model definition
            print("Detected old FEDformer architecture, using compatible model definition...")
            
            class PositionalEncodingOld(nn.Module):
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

            class FourierBlockOld(nn.Module):
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

            class FEDformerEncoderOld(nn.Module):
                def __init__(self, d_model, n_layers=3, modes=32, dropout=0.1):
                    super().__init__()
                    self.layers = nn.ModuleList([
                        FourierBlockOld(d_model, modes, dropout) for _ in range(n_layers)
                    ])
                def forward(self, x):
                    for layer in self.layers:
                        x = layer(x)
                    return x

            class FEDformerDecoderOld(nn.Module):
                def __init__(self, d_model, pred_len, modes=32, dropout=0.1):
                    super().__init__()
                    self.pred_len = pred_len
                    self.modes = modes
                    self.dropout = nn.Dropout(dropout)
                    self.norm = nn.LayerNorm(d_model)
                    self.cross_attention = nn.MultiheadAttention(
                        d_model, num_heads=8, dropout=dropout, batch_first=True
                    )
                    self.fourier_block = FourierBlockOld(d_model, modes, dropout)
                def forward(self, decoder_input, encoder_output):
                    attn_out, _ = self.cross_attention(
                        decoder_input, encoder_output, encoder_output
                    )
                    decoder_input = decoder_input + attn_out
                    x = self.fourier_block(decoder_input)
                    return x

            class FEDformerOld(nn.Module):
                def __init__(self, input_dim, output_dim, pred_len, 
                             d_model=256, n_encoder_layers=3, n_decoder_layers=2, 
                             modes=32, dropout=0.1):
                    super().__init__()
                    self.pred_len = pred_len
                    self.d_model = d_model
                    self.embed = nn.Linear(input_dim, d_model)
                    self.pos_encoder = PositionalEncodingOld(d_model)
                    self.encoder = FEDformerEncoderOld(d_model, n_encoder_layers, modes, dropout)
                    self.decoder_layers = nn.ModuleList([
                        FEDformerDecoderOld(d_model, pred_len, modes, dropout) 
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

            fed = FEDformerOld(
                input_dim=input_dim,
                output_dim=output_dim,
                pred_len=pred_len,
                d_model=256,
                n_encoder_layers=3,
                n_decoder_layers=2,
                modes=32,
                dropout=0.1
            ).cuda()
            fed.load_state_dict(state_dict)
            print("Loaded old FEDformer architecture (d_model=256, encoder_layers=3, decoder_layers=2)")
            print("NOTE: To use the improved architecture, please retrain the model with train_transformer.py")
    
    fed.eval()

    # -------------------------
    # EVALUATE ALL THREE MODELS
    # -------------------------
    bs = 512
    lstm_preds = []
    fed_preds = []
    tcnn_preds = []

    with torch.no_grad():
        for i in range(0, len(X_t), bs):
            lstm_preds.append(lstm(X_t[i:i+bs]).cpu().numpy())
            fed_preds.append(fed(X_t[i:i+bs]).cpu().numpy())
            tcnn_preds.append(tcnn(X_t[i:i+bs]).cpu().numpy())

    lstm_preds = np.concatenate(lstm_preds, axis=0)
    fed_preds  = np.concatenate(fed_preds,  axis=0)
    tcnn_preds = np.concatenate(tcnn_preds, axis=0)

    true_last = Y[:, -1, :]
    lstm_last = lstm_preds[:, -1, :]
    fed_last  = fed_preds[:, -1, :]
    tcnn_last = tcnn_preds[:, -1, :]

    print("\n" + "=" * 80)
    print("MODEL COMPARISON: LSTM vs FEDformer vs TCNN")
    print("=" * 80)
    print(f"{'Metric':<12} {'LSTM':<15} {'FEDformer':<15} {'TCNN':<15} {'Winner':<10}")
    print("-" * 80)
    
    lstm_mae = mean_absolute_error(true_last, lstm_last)
    fed_mae = mean_absolute_error(true_last, fed_last)
    tcnn_mae = mean_absolute_error(true_last, tcnn_last)
    
    lstm_mse = mean_squared_error(true_last, lstm_last)
    fed_mse = mean_squared_error(true_last, fed_last)
    tcnn_mse = mean_squared_error(true_last, tcnn_last)
    
    lstm_rmse = np.sqrt(lstm_mse)
    fed_rmse = np.sqrt(fed_mse)
    tcnn_rmse = np.sqrt(tcnn_mse)
    
    lstm_r2 = r2_score(true_last, lstm_last)
    fed_r2 = r2_score(true_last, fed_last)
    tcnn_r2 = r2_score(true_last, tcnn_last)
    
    metrics = [
        ("MAE", lstm_mae, fed_mae, tcnn_mae, "lower"),
        ("MSE", lstm_mse, fed_mse, tcnn_mse, "lower"),
        ("RMSE", lstm_rmse, fed_rmse, tcnn_rmse, "lower"),
        ("R²", lstm_r2, fed_r2, tcnn_r2, "higher"),
    ]
    
    for name, lstm_val, fed_val, tcnn_val, better in metrics:
        if better == "lower":
            vals = [lstm_val, fed_val, tcnn_val]
            winner_idx = np.argmin(vals)
            winner = ["LSTM", "FEDformer", "TCNN"][winner_idx]
        else:
            vals = [lstm_val, fed_val, tcnn_val]
            winner_idx = np.argmax(vals)
            winner = ["LSTM", "FEDformer", "TCNN"][winner_idx]
        
        print(f"{name:<12} {lstm_val:.6f}  {fed_val:.6f}  {tcnn_val:.6f}  {winner:<10}")

    # Calculate improvement percentages (relative to LSTM baseline)
    fed_mae_improvement = ((lstm_mae - fed_mae) / lstm_mae) * 100
    tcnn_mae_improvement = ((lstm_mae - tcnn_mae) / lstm_mae) * 100
    
    fed_rmse_improvement = ((lstm_rmse - fed_rmse) / lstm_rmse) * 100
    tcnn_rmse_improvement = ((lstm_rmse - tcnn_rmse) / lstm_rmse) * 100
    
    fed_r2_improvement = ((fed_r2 - lstm_r2) / abs(lstm_r2)) * 100 if lstm_r2 != 0 else 0
    tcnn_r2_improvement = ((tcnn_r2 - lstm_r2) / abs(lstm_r2)) * 100 if lstm_r2 != 0 else 0

    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS (vs LSTM baseline)")
    print("=" * 80)
    print(f"MAE Improvement:")
    print(f"  FEDformer: {fed_mae_improvement:+.2f}% ({'better' if fed_mae_improvement > 0 else 'worse'})")
    print(f"  TCNN:      {tcnn_mae_improvement:+.2f}% ({'better' if tcnn_mae_improvement > 0 else 'worse'})")
    print(f"RMSE Improvement:")
    print(f"  FEDformer: {fed_rmse_improvement:+.2f}% ({'better' if fed_rmse_improvement > 0 else 'worse'})")
    print(f"  TCNN:      {tcnn_rmse_improvement:+.2f}% ({'better' if tcnn_rmse_improvement > 0 else 'worse'})")
    print(f"R² Improvement:")
    print(f"  FEDformer: {fed_r2_improvement:+.2f}% ({'better' if fed_r2_improvement > 0 else 'worse'})")
    print(f"  TCNN:      {tcnn_r2_improvement:+.2f}% ({'better' if tcnn_r2_improvement > 0 else 'worse'})")

    # --------------------------------------------
    # CREATE COMPARISON VISUALIZATION
    # --------------------------------------------
    import matplotlib.pyplot as plt
    
    plot_dir = "/data/comparison_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Calculate additional metrics for plots
    lstm_errors = lstm_last - true_last
    fed_errors = fed_last - true_last
    tcnn_errors = tcnn_last - true_last
    
    lstm_r2_features = [r2_score(true_last[:, f], lstm_last[:, f]) for f in range(output_dim)]
    fed_r2_features = [r2_score(true_last[:, f], fed_last[:, f]) for f in range(output_dim)]
    tcnn_r2_features = [r2_score(true_last[:, f], tcnn_last[:, f]) for f in range(output_dim)]
    
    lstm_mae_features = [mean_absolute_error(true_last[:, f], lstm_last[:, f]) for f in range(output_dim)]
    fed_mae_features = [mean_absolute_error(true_last[:, f], fed_last[:, f]) for f in range(output_dim)]
    tcnn_mae_features = [mean_absolute_error(true_last[:, f], tcnn_last[:, f]) for f in range(output_dim)]
    
    # Calculate MAE over horizon for all models
    lstm_horizon_mae = []
    fed_horizon_mae = []
    tcnn_horizon_mae = []
    for t in range(pred_len):
        lstm_horizon_mae.append(mean_absolute_error(Y[:, t, :], lstm_preds[:, t, :]))
        fed_horizon_mae.append(mean_absolute_error(Y[:, t, :], fed_preds[:, t, :]))
        tcnn_horizon_mae.append(mean_absolute_error(Y[:, t, :], tcnn_preds[:, t, :]))
    
    # Feature names matching the target columns
    feature_names = ['Temperature', 'Relative Humidity', 'Wind Speed', 
                     'Wind Direction', 'Soil Temperature', 'Soil Moisture']
    features = feature_names[:output_dim]  # Ensure we have the right number
    
    hours = np.arange(1, pred_len + 1)
    sample_idx = len(lstm_preds) // 2
    
    # Create comprehensive comparison figure - larger to accommodate 6 feature plots
    fig = plt.figure(figsize=(28, 20))
    gs = fig.add_gridspec(5, 3, hspace=0.4, wspace=0.3)
    
    # 1-6. Forecast over time for each feature (6 separate plots in 2 rows)
    feature_axes = []
    for f in range(output_dim):
        row = f // 3
        col = f % 3
        ax = fig.add_subplot(gs[row, col])
        ax.plot(hours, Y[sample_idx, :, f], label='True', linewidth=2.5, alpha=0.8, color='black')
        ax.plot(hours, lstm_preds[sample_idx, :, f], label='LSTM', linewidth=2, linestyle='--', alpha=0.7, color='#1f77b4')
        ax.plot(hours, fed_preds[sample_idx, :, f], label='FEDformer', linewidth=2, linestyle=':', alpha=0.7, color='#ff7f0e')
        ax.plot(hours, tcnn_preds[sample_idx, :, f], label='TCNN', linewidth=2, linestyle='-.', alpha=0.7, color='#2ca02c')
        ax.set_xlabel('Forecast Hour', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(f'{feature_names[f]} - Forecast Over 168 Hours', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        feature_axes.append(ax)
    
    # 7-9. Scatter plots for all 3 models (row 2)
    colors = plt.cm.tab10(np.linspace(0, 1, output_dim))
    
    ax2 = fig.add_subplot(gs[2, 0])
    for f in range(output_dim):
        ax2.scatter(true_last[:, f], lstm_last[:, f], s=1, alpha=0.3, c=[colors[f]], label=feature_names[f])
    min_val = min(true_last.min(), lstm_last.min())
    max_val = max(true_last.max(), lstm_last.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('True Values', fontsize=10)
    ax2.set_ylabel('Predicted Values', fontsize=10)
    ax2.set_title('LSTM: Predicted vs True', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=7, ncol=2, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[2, 1])
    for f in range(output_dim):
        ax3.scatter(true_last[:, f], fed_last[:, f], s=1, alpha=0.3, c=[colors[f]], label=feature_names[f])
    min_val = min(true_last.min(), fed_last.min())
    max_val = max(true_last.max(), fed_last.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('True Values', fontsize=10)
    ax3.set_ylabel('Predicted Values', fontsize=10)
    ax3.set_title('FEDformer: Predicted vs True', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=7, ncol=2, loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    ax3_tcnn = fig.add_subplot(gs[2, 2])
    for f in range(output_dim):
        ax3_tcnn.scatter(true_last[:, f], tcnn_last[:, f], s=1, alpha=0.3, c=[colors[f]], label=feature_names[f])
    min_val = min(true_last.min(), tcnn_last.min())
    max_val = max(true_last.max(), tcnn_last.max())
    ax3_tcnn.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax3_tcnn.set_xlabel('True Values', fontsize=10)
    ax3_tcnn.set_ylabel('Predicted Values', fontsize=10)
    ax3_tcnn.set_title('TCNN: Predicted vs True', fontsize=12, fontweight='bold')
    ax3_tcnn.legend(fontsize=7, ncol=2, loc='upper left')
    ax3_tcnn.grid(True, alpha=0.3)
    
    # 10. Error histograms comparison - All 3 models (row 3)
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.hist(lstm_errors.flatten(), bins=50, alpha=0.5, label='LSTM', edgecolor='black', color='#1f77b4')
    ax4.hist(fed_errors.flatten(), bins=50, alpha=0.5, label='FEDformer', edgecolor='black', color='#ff7f0e')
    ax4.hist(tcnn_errors.flatten(), bins=50, alpha=0.5, label='TCNN', edgecolor='black', color='#2ca02c')
    ax4.axvline(0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Error', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Error Distribution Comparison', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 11. R² per feature comparison - All 3 models (row 3)
    ax5 = fig.add_subplot(gs[3, 1])
    x = np.arange(len(features))
    width = 0.25
    ax5.bar(x - width, lstm_r2_features, width, label='LSTM', color='#1f77b4', edgecolor='black')
    ax5.bar(x, fed_r2_features, width, label='FEDformer', color='#ff7f0e', edgecolor='black')
    ax5.bar(x + width, tcnn_r2_features, width, label='TCNN', color='#2ca02c', edgecolor='black')
    ax5.set_ylabel('R² Score', fontsize=10)
    ax5.set_title('R² Per Feature Comparison', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(features, rotation=45, ha='right')
    ax5.legend()
    ax5.set_ylim([0, 1])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 12. MAE per feature comparison - All 3 models (row 3)
    ax6 = fig.add_subplot(gs[3, 2])
    ax6.bar(x - width, lstm_mae_features, width, label='LSTM', color='#1f77b4', edgecolor='black')
    ax6.bar(x, fed_mae_features, width, label='FEDformer', color='#ff7f0e', edgecolor='black')
    ax6.bar(x + width, tcnn_mae_features, width, label='TCNN', color='#2ca02c', edgecolor='black')
    ax6.set_ylabel('MAE', fontsize=10)
    ax6.set_title('MAE Per Feature Comparison', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(features, rotation=45, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 13. MAE over forecast horizon comparison - All 3 models (row 4)
    ax7 = fig.add_subplot(gs[4, 0])
    ax7.plot(lstm_horizon_mae, label='LSTM', linewidth=2, color='#1f77b4')
    ax7.plot(fed_horizon_mae, label='FEDformer', linewidth=2, color='#ff7f0e')
    ax7.plot(tcnn_horizon_mae, label='TCNN', linewidth=2, color='#2ca02c')
    ax7.set_xlabel('Forecast Step (Hours)', fontsize=10)
    ax7.set_ylabel('MAE', fontsize=10)
    ax7.set_title('MAE over Forecast Horizon', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 14. Metrics comparison table - All 3 models (row 4)
    ax8 = fig.add_subplot(gs[4, 1])
    ax8.axis('off')
    
    # Determine winners for each metric
    mae_winner = ['LSTM', 'FEDformer', 'TCNN'][np.argmin([lstm_mae, fed_mae, tcnn_mae])]
    rmse_winner = ['LSTM', 'FEDformer', 'TCNN'][np.argmin([lstm_rmse, fed_rmse, tcnn_rmse])]
    r2_winner = ['LSTM', 'FEDformer', 'TCNN'][np.argmax([lstm_r2, fed_r2, tcnn_r2])]
    
    metrics_text = f"""
    METRICS COMPARISON
    
    MAE:
      LSTM:      {lstm_mae:.6f}
      FEDformer: {fed_mae:.6f}
      TCNN:      {tcnn_mae:.6f}
      Winner:    {mae_winner}
    
    RMSE:
      LSTM:      {lstm_rmse:.6f}
      FEDformer: {fed_rmse:.6f}
      TCNN:      {tcnn_rmse:.6f}
      Winner:    {rmse_winner}
    
    R²:
      LSTM:      {lstm_r2:.6f}
      FEDformer: {fed_r2:.6f}
      TCNN:      {tcnn_r2:.6f}
      Winner:    {r2_winner}
    """
    ax8.text(0.05, 0.5, metrics_text, fontsize=9, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 15. Improvement percentages and winner (row 4, column 2)
    ax9 = fig.add_subplot(gs[4, 2])
    ax9.axis('off')
    
    # Count wins
    lstm_wins = sum([lstm_mae == min([lstm_mae, fed_mae, tcnn_mae]),
                     lstm_rmse == min([lstm_rmse, fed_rmse, tcnn_rmse]),
                     lstm_r2 == max([lstm_r2, fed_r2, tcnn_r2])])
    fed_wins = sum([fed_mae == min([lstm_mae, fed_mae, tcnn_mae]),
                    fed_rmse == min([lstm_rmse, fed_rmse, tcnn_rmse]),
                    fed_r2 == max([lstm_r2, fed_r2, tcnn_r2])])
    tcnn_wins = sum([tcnn_mae == min([lstm_mae, fed_mae, tcnn_mae]),
                     tcnn_rmse == min([lstm_rmse, fed_rmse, tcnn_rmse]),
                     tcnn_r2 == max([lstm_r2, fed_r2, tcnn_r2])])
    
    wins = [lstm_wins, fed_wins, tcnn_wins]
    winner_idx = np.argmax(wins)
    winner_name = ['LSTM', 'FEDformer', 'TCNN'][winner_idx]
    winner_color = ['lightcoral', 'lightgreen', 'lightyellow'][winner_idx]
    
    combined_text = f"""
    IMPROVEMENT vs LSTM
    
    MAE:
      FEDformer: {fed_mae_improvement:+.2f}%
      TCNN:      {tcnn_mae_improvement:+.2f}%
    
    RMSE:
      FEDformer: {fed_rmse_improvement:+.2f}%
      TCNN:      {tcnn_rmse_improvement:+.2f}%
    
    R²:
      FEDformer: {fed_r2_improvement:+.2f}%
      TCNN:      {tcnn_r2_improvement:+.2f}%
    
    {'='*22}
    OVERALL WINNER
    
    LSTM:      {lstm_wins}/3
    FEDformer: {fed_wins}/3
    TCNN:      {tcnn_wins}/3
    
    WINNER: {winner_name}
    {'='*22}
    """
    ax9.text(0.05, 0.5, combined_text, fontsize=9, family='monospace', fontweight='bold',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor=winner_color, alpha=0.7))
    
    # Add overall title
    fig.suptitle('LSTM vs FEDformer vs TCNN: Complete Model Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    # Save combined comparison plot
    plt.savefig(f"{plot_dir}/model_comparison_complete.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved comparison plot: {plot_dir}/model_comparison_complete.png")

@app.local_entrypoint()
def main():
    print("Starting model comparison...")
    compare_models.remote()
    print("Comparison completed!")

