import modal

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
    import torch
    import torch.nn as nn
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    print("Files available in /data:", os.listdir("/data"))

    # Make plot directory
    plot_dir = "/data/plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Load data
    X = np.load("/data/X.npy")
    Y = np.load("/data/Y.npy")

    X_t = torch.tensor(X, dtype=torch.float32).cuda()

    input_dim = X.shape[2]
    output_dim = Y.shape[2]
    pred_len = Y.shape[1]

    # -------------------------
    # Model definitions
    # -------------------------
    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=128, layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True)

        def forward(self, x):
            _, (h, c) = self.lstm(x)
            return h, c

    class Decoder(nn.Module):
        def __init__(self, output_dim, hidden_dim=128, layers=2, seq_len=168):
            super().__init__()
            self.seq_len = seq_len
            self.lstm = nn.LSTM(output_dim, hidden_dim, layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, h, c):
            batch_size = h.shape[1]
            zeros = torch.zeros((batch_size, self.seq_len, output_dim), device=h.device)
            out, _ = self.lstm(zeros, (h, c))
            return self.fc(out)

    class Seq2Seq(nn.Module):
        def __init__(self, input_dim, output_dim, pred_len):
            super().__init__()
            self.encoder = Encoder(input_dim)
            self.decoder = Decoder(output_dim, seq_len=pred_len)

        def forward(self, x):
            h, c = self.encoder(x)
            return self.decoder(h, c)

    # -------------------------
    # Load model
    # -------------------------
    model = Seq2Seq(input_dim, output_dim, pred_len).cuda()
    model.load_state_dict(torch.load("/data/lstm_weather_model.pth"))
    model.eval()

    print("Model loaded successfully.")

    # -------------------------
    # Batched inference
    # -------------------------
    batch_size = 512
    preds_list = []

    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i:i + batch_size]
            out = model(batch)
            preds_list.append(out.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)

    # Last-step metrics
    true_last = Y[:, -1, :]
    pred_last = preds[:, -1, :]

    mae = mean_absolute_error(true_last, pred_last)
    mse = mean_squared_error(true_last, pred_last)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_last, pred_last)

    print("\n===== FINAL MODEL PERFORMANCE =====")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2:", r2)

    # --------------------------------------------
    # SAVE PLOTS
    # --------------------------------------------

    features = [f"Feature {i}" for i in range(output_dim)]

    # 1. Predicted vs True (last timestep)
    for f in range(output_dim):
        plt.figure(figsize=(6, 4))
        plt.scatter(true_last[:, f], pred_last[:, f], s=2, alpha=0.5)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Last-step Prediction vs True — {features[f]}")
        plt.savefig(f"{plot_dir}/scatter_feature_{f}.png")
        plt.close()

    # 2. Error histogram
    errors = pred_last - true_last
    plt.figure(figsize=(6, 4))
    plt.hist(errors.flatten(), bins=50)
    plt.title("Error Distribution (Last Step)")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.savefig(f"{plot_dir}/error_hist.png")
    plt.close()

    # 3. R² per feature
    r2_features = []
    for f in range(output_dim):
        r2_f = r2_score(true_last[:, f], pred_last[:, f])
        r2_features.append(r2_f)

    plt.figure(figsize=(6, 4))
    plt.bar(features, r2_features)
    plt.title("R² Per Feature (Last Step)")
    plt.ylabel("R²")
    plt.xticks(rotation=45)
    plt.savefig(f"{plot_dir}/r2_per_feature.png")
    plt.close()

    # 4. MAE across forecast horizon
    horizon_mae = []
    for t in range(pred_len):
        horizon_mae.append(mean_absolute_error(Y[:, t, :], preds[:, t, :]))

    plt.figure(figsize=(8, 4))
    plt.plot(horizon_mae)
    plt.title("MAE over Forecast Horizon (1 → 168 hours)")
    plt.xlabel("Forecast Step")
    plt.ylabel("MAE")
    plt.savefig(f"{plot_dir}/mae_over_horizon.png")
    plt.close()

    print("\nSaved plot files in:", plot_dir)
    print("Evaluation and graphing completed.")
