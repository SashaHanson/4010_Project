import modal
import os

app = modal.App("lstm-advanced-training")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "numpy",
    )
)

volume = modal.Volume.from_name("dataset")

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    volumes={"/data": volume},
)
def train_lstm_advanced():
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # ======================================
    # Load X.npy and Y.npy
    # ======================================
    print("Files in /data:", os.listdir("/data"))

    X = np.load("/data/X.npy")
    Y = np.load("/data/Y.npy")

    print("Loaded X:", X.shape)
    print("Loaded Y:", Y.shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    # Shapes
    num_samples = X.shape[0]
    input_len = X.shape[1]
    input_dim = X.shape[2]
    pred_len = Y.shape[1]
    output_dim = Y.shape[2]

    print(f"Input length={input_len}, Features={input_dim}, PredLen={pred_len}, Outputs={output_dim}")

    # ======================================
    # Train/Val/Test split
    # ======================================
    train_end = int(num_samples * 0.7)
    val_end = int(num_samples * 0.85)

    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=64, shuffle=False)

    # ======================================
    # MODEL DEFINITIONS
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

    # ======================================
    # Instantiate model
    # ======================================
    encoder = Encoder(input_dim)
    decoder = Decoder(output_dim)
    model = Seq2Seq(encoder, decoder, output_dim, pred_len).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.MSELoss()

    # ======================================
    # Create checkpoint dir
    # ======================================
    ckpt_dir = "/data/checkpoints"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # ======================================
    # TRAINING LOOP (with checkpoints)
    # ======================================
    EPOCHS = 40
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            preds = model(bx, by, teacher_forcing_ratio=0.6)
            loss = criterion(preds, by)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * bx.size(0)

        train_loss /= len(train_loader.dataset)

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx, by, teacher_forcing_ratio=0)
                loss = criterion(preds, by)
                val_loss += loss.item() * bx.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        # -------------------------
        # SAVE PER-EPOCH CHECKPOINT
        # -------------------------
        epoch_ckpt = f"{ckpt_dir}/lstm_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_ckpt)
        print(f"Saved epoch checkpoint → {epoch_ckpt}")

        # Always save a "latest" checkpoint
        torch.save(model.state_dict(), "/data/lstm_weather_model_latest.pth")

        # -------------------------
        # BEST MODEL
        # -------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            torch.save(best_state, "/data/lstm_weather_model_advanced.pth")
            print("Updated BEST model → /data/lstm_weather_model_advanced.pth")

    # ======================================
    # FINAL TEST
    # ======================================
    model.load_state_dict(best_state)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            preds = model(bx, by, teacher_forcing_ratio=0)
            loss = criterion(preds, by)
            test_loss += loss.item() * bx.size(0)

    test_loss /= len(test_loader.dataset)

    print("Final Test Loss:", test_loss)
