import modal

app = modal.App("tcnn-training")

# Use an A100 GPU + PyTorch in the image
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
)

# Use your existing dataset volume
volume = modal.Volume.from_name("dataset")


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,  # 1 hour
    volumes={"/data": volume},
)
def train_tcnn():
    import torch
    import torch.nn as nn
    import numpy as np
    import os

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)
    print("Files in /data:", os.listdir("/data"))

    # ---------------------------
    # Load data from Modal volume
    # ---------------------------
    X = np.load("/data/X.npy")
    Y = np.load("/data/Y.npy")

    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = X.shape[2]      # number of features per timestep
    output_dim = Y.shape[2]     # future feature dims
    pred_len = Y.shape[1]       # 168 future timesteps

    # ---------------------------
    # Temporal CNN model
    # ---------------------------
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
            # x: (batch, seq_len, features)
            x = x.transpose(1, 2)  # to (batch, features, seq_len)
            features = self.tcn(x)
            last_step = features[:, :, -1]  # causal last timestep representation
            out = self.head(last_step)
            return out.view(-1, pred_len, output_dim)

    model = TCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # ---------------------------
    # Training loop
    # ---------------------------
    EPOCHS = 5  # Increase later after confirming correctness

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for bx, by in loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS}  Loss: {total_loss:.4f}")

    # ---------------------------
    # Save model to the volume
    # ---------------------------
    save_path = "/data/tcnn_weather_model.pth"
    torch.save(model.state_dict(), save_path)

    print("Saved model to:", save_path)
