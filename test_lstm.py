import modal
import os
import shutil

app = modal.App("lstm-eval")

#This Modal app loads the preprocessed weather windows (X, Y) and a trained
#LSTM model from the shared Modal volume. It runs the model in evaluation
#mode over the full dataset, computes regression metrics, and writes plots
#that show how well the network forecasts the next 168 hours. This was done so that I can
#eveluate how my model is doing so that I know whether I am ready to submit it and compare 
#it to the other models

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "tqdm",
    )
)

#this is where the app is going to save its information
volume = modal.Volume.from_name("dataset")

#for this app we are going to use an A100 GPU and run for at most 30 minutes
@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 30,
    volumes={"/data": volume},
)
#these are the important imports that are needed to run this code
def evaluate_lstm():
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from tqdm import tqdm

    print("Files available in /data:", os.listdir("/data"))

    # --------- Clean existing plot directory ----------
    #this is very important as I don't want my old plots to be there when I run the app again
    plot_dir = "/data/plots_lstm_advanced"
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)

    # --------- Load data ----------
    #this is where we load the data from the modal volume
    X = np.load("/data/X.npy")
    Y = np.load("/data/Y.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    # Recover the same dimensional information used during training:
    # - input_dim: number of input features per hour in X
    # - output_dim: number of target features per hour in Y
    # - pred_len: how many hours into the future we forecast (168‑step horizon)
    input_dim = X.shape[2]
    output_dim = Y.shape[2]
    pred_len = Y.shape[1]

    # ======================================
    # MODEL DEFINITIONS – EXACT COPY OF TRAIN
    # ======================================
    # Same encoder–decoder with attention as in train_lstm:
    # - Encoder: encodes the past window into hidden states for each timestep.
    # - Attention: scores how relevant each past timestep is for the current prediction.
    # - Decoder: generates the 168‑step forecast one hour at a time.
    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=256, layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0,
            )

        def forward(self, x):
            # x: [batch, input_len, input_dim]
            # returns all timestep hidden states plus the final (h, c)
            outputs, (h, c) = self.lstm(x)
            return outputs, (h, c)

    class Attention(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, hidden, encoder_outputs):
            # hidden: last decoder hidden state (query)
            # encoder_outputs: all encoder states over time (keys/values)
            # returns attention weights over timesteps [batch, src_len]
            seq_len = encoder_outputs.size(1)
            hidden_exp = hidden.unsqueeze(1).repeat(1, seq_len, 1)
            energy = torch.tanh(self.attn(torch.cat((hidden_exp, encoder_outputs), dim=2)))
            attention = self.v(energy).squeeze(2)
            return torch.softmax(attention, dim=1)

    class Decoder(nn.Module):
        def __init__(self, output_dim, hidden_dim=256, layers=2, dropout=0.3):
            super().__init__()
            self.attention = Attention(hidden_dim)
            self.lstm = nn.LSTM(
                output_dim + hidden_dim,
                hidden_dim,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0,
            )
            self.fc_out = nn.Linear(hidden_dim, output_dim)

        def forward(self, y_prev, hidden, cell, encoder_outputs):
            # One decoding step: use attention to build a context vector over
            # all encoder states, then feed [y_prev, context] through the LSTM
            # to produce the next hour's prediction.
            attn_weights = self.attention(hidden[-1], encoder_outputs)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

            lstm_input = torch.cat((y_prev, context), dim=1).unsqueeze(1)
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            pred = self.fc_out(output.squeeze(1))
            return pred, hidden, cell

    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder, output_dim, pred_len):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.pred_len = pred_len
            self.output_dim = output_dim

        def forward(self, x, y=None, teacher_forcing_ratio=0.5):
            # Full seq2seq forward pass. During evaluation we call this with
            # y=None and teacher_forcing_ratio=0 so the decoder always feeds
            # back its own predictions (pure autoregressive forecasting).
            encoder_outputs, (h, c) = self.encoder(x)
            batch = x.size(0)
            outputs = torch.zeros(batch, self.pred_len, self.output_dim, device=x.device)

            y_prev = torch.zeros(batch, self.output_dim, device=x.device)

            for t in range(self.pred_len):
                out, h, c = self.decoder(y_prev, h, c, encoder_outputs)
                outputs[:, t] = out

                if y is not None and torch.rand(1).item() < teacher_forcing_ratio:
                    y_prev = y[:, t]
                else:
                    y_prev = out

            return outputs

    # --------- Load checkpoint and extract config ----------
    #this is where we load the weights from the completed training
    checkpoint_path = "/data/lstm_weather_model_advanced.pth"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Please run train_lstm.py first to train and save the model."
        )
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {e}")
    
    # Handle both old format (just state_dict) and new format (full checkpoint dict)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # New format: full checkpoint
        model_state = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
        best_epoch = checkpoint.get("epoch", "unknown")
        best_val_loss = checkpoint.get("best_val_loss", "unknown")
        print(f"✓ Loaded checkpoint from epoch {best_epoch} with val loss: {best_val_loss:.6f}")
        if config:
            print(f"  Model config: hidden_dim={config.get('hidden_dim', 256)}, "
                  f"layers={config.get('num_layers', 2)}, dropout={config.get('dropout', 0.3)}")
    else:
        # Old format: just state_dict
        model_state = checkpoint
        config = {}
        print("✓ Loaded checkpoint (old format - using default config)")
    
    #Extract hyperparameters from config or use defaults,
    #this is important so that we can instantiate the model with the same 
    #architecture as the trained model
    hidden_dim = config.get("hidden_dim", 256)
    num_layers = config.get("num_layers", 2)
    dropout = config.get("dropout", 0.3)
    
    # --------- Instantiate model with matching architecture ----------
    #again we want to instantiate the model with the same architecture as the trained model, since 
    #without this we will get errors or not very accurate testing results
    encoder = Encoder(input_dim, hidden_dim=hidden_dim, layers=num_layers, dropout=dropout)
    decoder = Decoder(output_dim, hidden_dim=hidden_dim, layers=num_layers, dropout=dropout)
    model = Seq2Seq(encoder, decoder, output_dim, pred_len).to(device)
    
    # Load weights
    model.load_state_dict(model_state)
    model.eval()

    print("Model loaded successfully.")

    # ======================================
    # Batched inference (no teacher forcing)
    # ======================================
    batch_size = 512
    preds_list = []
    use_amp = config.get("use_amp", True) and device == "cuda"

    print(f"Running inference with batch_size={batch_size}, use_amp={use_amp}...")
    with torch.no_grad():
        for i in tqdm(range(0, len(X_t), batch_size), desc="Inference"):
            batch = X_t[i : i + batch_size]
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(batch, y=None, teacher_forcing_ratio=0.0)
            else:
                out = model(batch, y=None, teacher_forcing_ratio=0.0)
            
            preds_list.append(out.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    print(f"Inference complete. Predictions shape: {preds.shape}")

    # ------------------------------------
    # METRICS (using entire dataset)
    # ------------------------------------
    #For the scalar summary metrics we only look at the very last forecast step
    #of each sequence: true_last are the ground‑truth values at horizon 168,
    #pred_last are the model’s predictions at that same final hour. The MAE/MSE/RMSE
    #and R² we compute next all compare these two arrays.
    true_last = Y[:, -1, :]
    pred_last = preds[:, -1, :]

    mae = mean_absolute_error(true_last, pred_last)
    mse = mean_squared_error(true_last, pred_last)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_last, pred_last)

    print("\n===== FINAL MODEL PERFORMANCE (Last Step) =====")
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R²  :", r2)

    target_features = [
        "Temperature",
        "Relative Humidity",
        "Wind Speed",
        "Wind Direction",
        "Soil Temperature",
        "Soil Moisture",
    ]

    # ------------ 1. MAE per feature ------------
    mae_per_feature = [
        mean_absolute_error(true_last[:, f], pred_last[:, f])
        for f in range(output_dim)
    ]

    plt.figure(figsize=(6, 4))
    plt.bar(target_features, mae_per_feature)
    plt.title("MAE Per Feature (Last Step)")
    plt.ylabel("MAE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/mae_per_feature.png")
    plt.close()

    # ------------ 2. Error histogram ------------
    errors = pred_last - true_last
    plt.figure(figsize=(6, 4))
    plt.hist(errors.flatten(), bins=50)
    plt.title("Error Distribution (Last Step)")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/error_hist.png")
    plt.close()

    # ------------ 3. R² per feature ------------
    r2_features = [r2_score(true_last[:, f], pred_last[:, f]) for f in range(output_dim)]

    plt.figure(figsize=(6, 4))
    plt.bar(target_features, r2_features)
    plt.title("R² Per Feature (Last Step)")
    plt.ylabel("R²")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/r2_per_feature.png")
    plt.close()

    # ------------ 4. MAE across horizon ------------
    horizon_mae = [
        mean_absolute_error(Y[:, t, :], preds[:, t, :])
        for t in range(pred_len)
    ]

    plt.figure(figsize=(8, 4))
    plt.plot(horizon_mae)
    plt.title("MAE over Forecast Horizon (1 → 168 hours)")
    plt.xlabel("Forecast Step")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/mae_over_horizon.png")
    plt.close()

    # ------------ 5. Actual vs Predicted Series for each target ------------
    sample_idx = 300  #choose a random sample
    true_series = Y[sample_idx]
    pred_series = preds[sample_idx]

    for i, feature in enumerate(target_features):
        plt.figure(figsize=(10, 4))
        plt.plot(true_series[:, i], label="True")
        plt.plot(pred_series[:, i], label="Predicted")
        plt.title(f"Actual vs Predicted — {feature} (168h)")
        plt.xlabel("Forecast Horizon (hours)")
        plt.ylabel(feature)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/actual_vs_pred_{feature}.png")
        plt.close()

    print("\nSaved plot files in:", plot_dir)
    print("Evaluation and graphing completed.")
