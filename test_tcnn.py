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
        "tqdm",
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
    from tqdm import tqdm

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
        for i in tqdm(range(0, len(X_t), batch_size), desc="Inference batches", leave=False):
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
    for f in tqdm(range(output_dim), desc="Scatter plots", leave=False):
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
    for f in tqdm(range(output_dim), desc="R2 per feature", leave=False):
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
    for t in tqdm(range(pred_len), desc="MAE horizon", leave=False):
        horizon_mae.append(mean_absolute_error(Y[:, t, :], preds[:, t, :]))

    plt.figure(figsize=(8, 4))
    plt.plot(horizon_mae)
    plt.title("MAE over Forecast Horizon")
    plt.xlabel("Forecast Step")
    plt.ylabel("MAE")
    plt.savefig(f"{plot_dir}/mae_over_horizon.png")
    plt.close()

    # 5. Training loss per epoch (if available from training run)
    loss_history_path = "/data/tcnn_epoch_losses.npy"
    if os.path.exists(loss_history_path):
        try:
            epoch_losses = np.load(loss_history_path)
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
            plt.title("Training Loss per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/loss_per_epoch.png")
            plt.close()
            print("Saved loss curve:", f"{plot_dir}/loss_per_epoch.png")
        except Exception as e:
            print("Could not load loss history:", e)
    else:
        print("Loss history file not found; skipping loss plot.")

    # 6. Actual vs predicted curves for all targets on one sample
    sample_idx = 0
    if len(preds) > sample_idx:
        steps = np.arange(pred_len)
        fig, axes = plt.subplots(3, 2, figsize=(10, 8))
        axes = axes.flatten()
        for f in range(output_dim):
            ax = axes[f]
            ax.plot(steps, Y[sample_idx, :, f], label="Actual", linewidth=2)
            ax.plot(steps, preds[sample_idx, :, f], label="Predicted", linewidth=2, linestyle="--")
            ax.set_title(features[f])
            ax.set_xlabel("Forecast Step")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle="--", alpha=0.3)
        axes[0].legend()
        plt.tight_layout()
        curves_path = f"{plot_dir}/actual_vs_pred_sample{sample_idx}.png"
        plt.savefig(curves_path)
        plt.close()
        print("Saved target curves plot:", curves_path)
    else:
        print("Not enough samples to plot actual vs predicted curves.")

    print("\nSaved plot files in:", plot_dir)
    print("Evaluation and graphing completed.")
