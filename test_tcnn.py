import modal

app = modal.App("tcnn-eval")

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
def evaluate_tcnn():
    import torch
    import torch.nn as nn
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Files available in /data:", os.listdir("/data"))

    # Make plot directory
    plot_dir = "/data/plots_tcnn"
    os.makedirs(plot_dir, exist_ok=True)

    # Load data
    X = np.load("/data/X.npy")
    Y = np.load("/data/Y.npy")

    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    input_dim = X.shape[2]
    output_dim = Y.shape[2]
    pred_len = Y.shape[1]

    # -------------------------
    # Model definitions
    # -------------------------
    class TemporalConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
            super().__init__()
            padding = (kernel_size - 1) * dilation  # causal padding
            self.pad1 = nn.ConstantPad1d((padding, 0), 0)
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)

            self.pad2 = nn.ConstantPad1d((padding, 0), 0)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)

            self.residual = (
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels
                else None
            )

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
            channels = [input_dim, 64, 64, 64]
            layers = []
            for i in range(len(channels) - 1):
                layers.append(
                    TemporalConvBlock(
                        channels[i],
                        channels[i + 1],
                        kernel_size=3,
                        dilation=2 ** i,
                        dropout=0.1,
                    )
                )
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Linear(channels[-1], pred_len * output_dim)

        def forward(self, x):
            x = x.transpose(1, 2)
            features = self.tcn(x)
            last_step = features[:, :, -1]
            out = self.head(last_step)
            return out.view(-1, pred_len, output_dim)

    # -------------------------
    # Load model
    # -------------------------
    model = TCNN().to(device)
    state = torch.load("/data/tcnn_weather_model.pth", map_location=device)
    model.load_state_dict(state)
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
        plt.title(f"Last-step Prediction vs True - {features[f]}")
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

    # 3. R2 per feature
    r2_features = []
    for f in range(output_dim):
        r2_f = r2_score(true_last[:, f], pred_last[:, f])
        r2_features.append(r2_f)

    plt.figure(figsize=(6, 4))
    plt.bar(features, r2_features)
    plt.title("R2 Per Feature (Last Step)")
    plt.ylabel("R2")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/r2_per_feature.png")
    plt.close()

    # 4. MAE across forecast horizon
    horizon_mae = []
    for t in range(pred_len):
        horizon_mae.append(mean_absolute_error(Y[:, t, :], preds[:, t, :]))

    plt.figure(figsize=(8, 4))
    plt.plot(horizon_mae)
    plt.title("MAE over Forecast Horizon")
    plt.xlabel("Forecast Step")
    plt.ylabel("MAE")
    plt.savefig(f"{plot_dir}/mae_over_horizon.png")
    plt.close()

    print("\nSaved plot files in:", plot_dir)
    print("Evaluation and graphing completed.")
