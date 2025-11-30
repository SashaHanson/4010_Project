import modal
import os
import shutil

app = modal.App("lstm-eval")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "pandas",
    )
)

volume = modal.Volume.from_name("dataset")


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 30,
    volumes={"/data": volume},
)
def evaluate_lstm():
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    print("Files available in /data:", os.listdir("/data"))

    # --------- Clean existing plot directory ----------
    plot_dir = "/data/plots_lstm_advanced"
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)

    # --------- Load data ----------
    X = np.load("/data/X.npy")
    Y = np.load("/data/Y.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    input_dim = X.shape[2]
    output_dim = Y.shape[2]
    pred_len = Y.shape[1]

    # ======================================
    # MODEL DEFINITIONS – EXACT COPY OF TRAIN
    # ======================================
    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=256, layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=layers,
                batch_first=True,
                dropout=dropout,
            )

        def forward(self, x):
            outputs, (h, c) = self.lstm(x)
            return outputs, (h, c)

    class Attention(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, hidden, encoder_outputs):
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
                dropout=dropout,
            )
            self.fc_out = nn.Linear(hidden_dim, output_dim)

        def forward(self, y_prev, hidden, cell, encoder_outputs):
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

    # --------- Instantiate + load weights ----------
    encoder = Encoder(input_dim)
    decoder = Decoder(output_dim)
    model = Seq2Seq(encoder, decoder, output_dim, pred_len).to(device)

    state = torch.load("/data/lstm_weather_model_advanced.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("Model loaded successfully.")

    # ======================================
    # Batched inference (no teacher forcing)
    # ======================================
    batch_size = 512
    preds_list = []

    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i : i + batch_size]
            out = model(batch, y=None, teacher_forcing_ratio=0.0)
            preds_list.append(out.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)

    # ------------------------------------
    # METRICS (using entire dataset)
    # ------------------------------------
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
    sample_idx = 300  # choose a random sample
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
