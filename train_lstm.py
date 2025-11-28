import modal

app = modal.App("lstm-training")

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
    volumes={"/data": volume}
)
def train_lstm():
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
    # Seq2Seq model
    # ---------------------------
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
            batch = h.size(1)
            dec_input = torch.zeros((batch, self.seq_len, output_dim), device=h.device)
            out, _ = self.lstm(dec_input, (h, c))
            return self.fc(out)

    class Seq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder(input_dim)
            self.decoder = Decoder(output_dim)

        def forward(self, x):
            h, c = self.encoder(x)
            return self.decoder(h, c)

    model = Seq2Seq().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # ---------------------------
    # Training loop
    # ---------------------------
    EPOCHS = 5  # Increase later after confirming correctness

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for bx, by in loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss:.4f}")

    # ---------------------------
    # Save model to the volume
    # ---------------------------
    save_path = "/data/lstm_weather_model.pth"
    torch.save(model.state_dict(), save_path)

    print("Saved model to:", save_path)
