import modal

# Simple Modal app that loads preprocessed weather windows,
# trains a TCN on GPU, and writes the weights back to the same volume.
# Assumes Data_Preprocessing has already produced X.npy and Y.npy in the volume.
app = modal.App("tcnn-training")

# Use an A100 GPU + PyTorch in the image
# (Modal will place the container on GPU hardware when scheduled.)
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "tqdm")
)

# Reuse the dataset volume that already holds X.npy and Y.npy
volume = modal.Volume.from_name("dataset")


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,  # 1 hour
    volumes={"/data": volume},
)
def train_tcnn():
    """
    Load the sliding-window tensors from the shared volume, train a small
    Temporal Convolutional Network (TCN) to forecast the next 168 time steps
    from the previous 240, then persist the weights to the same volume.
    """
    import torch
    import torch.nn as nn
    import numpy as np
    from tqdm import tqdm
    import os

    # Prefer GPU (A100) when available; CPU is only a fallback for local runs.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)
    print("Files in /data:", os.listdir("/data"))

    # ---------------------------
    # Load data from Modal volume
    # ---------------------------
    # X: (num_samples, 240, 16)  past 240 hours of all scaled features
    # Y: (num_samples, 168, 6)   next 168 hours of scaled targets
    # Both are already standardized by the preprocessing step.
    X = np.load("/data/X.npy")
    Y = np.load("/data/Y.npy")

    # Move to torch tensors early so the dataloader stays on the chosen device.
    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)

    # Basic TensorDataset / DataLoader pairing for shuffled mini-batches.
    # Shuffle=True randomizes sequences so each epoch sees a different ordering,
    # which improves generalization and optimizer stability.
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    #change for sasha
    # Capture dimensions directly from the data to keep the model flexible.
    input_dim = X.shape[2]      # number of features per timestep (16)
    output_dim = Y.shape[2]     # number of targets per timestep (6)
    pred_len = Y.shape[1]       # how many future timesteps to predict (168)

    # ---------------------------
    # Temporal CNN model
    # ---------------------------
    class TemporalConvBlock(nn.Module):
        """
        A causal residual block with two dilated 1D convolutions.
        Padding on the left keeps sequence length unchanged while ensuring
        each output step only sees current/past inputs (no leakage from future).
        Dropout provides a bit of regularization for this small dataset size.
        """
        def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
            super().__init__()
            padding = (kernel_size - 1) * dilation  # causal padding for dilation
            self.pad1 = nn.ConstantPad1d((padding, 0), 0)
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)

            self.pad2 = nn.ConstantPad1d((padding, 0), 0)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)

            # Match channel dimensions so the residual can be added safely.
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

        def forward(self, x):
            # Left-pad then convolve so that position t only depends on <= t inputs.
            out = self.pad1(x)
            out = self.conv1(out)
            out = self.relu1(out)
            out = self.dropout1(out)

            out = self.pad2(out)
            out = self.conv2(out)
            out = self.relu2(out)
            out = self.dropout2(out)

            res = x if self.residual is None else self.residual(x)
            return out + res  # residual connection stabilizes training

    class TCNN(nn.Module):
        """
        Stacks several TemporalConvBlocks with doubling dilation (1, 2, 4, ...)
        to grow the causal receptive field over the 240-step history. Seven
        blocks with kernel=3 and dilations up to 64 give a receptive field
        wide enough (~255 steps) to cover the full window.
        """
        def __init__(self):
            super().__init__()
            channels = [input_dim, 128, 128, 128, 128, 128, 128, 128]  # 7 blocks
            layers = []
            for i in range(len(channels) - 1):
                layers.append(
                    TemporalConvBlock(
                        channels[i],
                        channels[i + 1],
                        kernel_size=3,
                        dilation=2 ** i,  # 1,2,4,8,16,32,64
                        dropout=0.1,
                    )
                )
            self.tcn = nn.Sequential(*layers)
            # Linear head maps the final feature vector to the flattened horizon.
            self.head = nn.Linear(channels[-1], pred_len * output_dim)

        def forward(self, x):
            # x: (batch, seq_len, features) -> transpose to (batch, channels, seq_len)
            x = x.transpose(1, 2)
            features = self.tcn(x)
            last_step = features[:, :, -1]  # causal last timestep representation
            out = self.head(last_step)      # flatten all future steps and targets
            return out.view(-1, pred_len, output_dim)  # reshape back to (B, 168, 6)

    model = TCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # ---------------------------
    # Training loop
    # ---------------------------
    EPOCHS = 50  # Keep short for quick checks; increase once pipeline is validated. _-------------------------------------->>>>>>>>> CHeck AFTER PLS!!!
    save_path = "/data/tcnn_weather_model.pth"  # overwritten each epoch; latest weights for eval
    checkpoint_path = "/data/tcnn_resume.pth"    # holds model + optimizer for resume
    loss_history_path = "/data/tcnn_epoch_losses.npy"

    # If a checkpoint exists, resume from it.
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resuming from epoch {start_epoch}")
    epoch_losses = []
    if os.path.exists(loss_history_path):
        try:
            epoch_losses = np.load(loss_history_path).tolist()
        except Exception as e:
            print("Warning: could not load previous loss history:", e)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0

        # tqdm renders a live bar over batches for this epoch.
        for bx, by in tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False):
            optimizer.zero_grad()
            preds = model(bx)           # forward pass (predict entire 168-step horizon)
            loss = criterion(preds, by) # compare full 168-step sequences
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS}  Avg Loss: {avg_loss:.4f}")

        # Persist a resume checkpoint and the latest weights each epoch.
        torch.save(
            {
                "epoch": epoch + 1,  # next epoch to run
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        torch.save(model.state_dict(), save_path)
        np.save(loss_history_path, np.array(epoch_losses))
        print("Saved checkpoint:", checkpoint_path)
        print("Saved weights:", save_path)
        print("Saved loss history:", loss_history_path)

    # ---------------------------
    # Save model to the volume
    # ---------------------------
    # Already saved on every epoch; final path kept for clarity.
    print("Training complete. Latest weights kept at:", save_path)
