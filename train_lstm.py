import modal
import os

app = modal.App("lstm-advanced-training")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "numpy",
        "tqdm",
    )
)

volume = modal.Volume.from_name("dataset")


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    volumes={"/data": volume}
)
def train_lstm_advanced():
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    import logging, json

    # ==========================
    # CONFIG
    # ==========================
    config = {
        "seed": 42,
        "batch_size": 64,
        "epochs": 40,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.3,
        "teacher_forcing_ratio": 0.6,
        "grad_clip": 1.0,
        "train_split": 0.7,
        "val_split": 0.85,
        "use_amp": True,
        "scheduler_type": "cosine",
        "scheduler_T_max": 40,
        "save_every_n_epochs": 5,
        "log_every_n_batches": 50,  # <--- moderate verbosity
    }

    # ==========================
    # LOGGING
    # ==========================
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting LSTM advanced training...")

    # ==========================
    # SEEDING
    # ==========================
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(config["seed"])
    logger.info(f"Random seed set → {config['seed']}")

    # ==========================
    # LOAD DATA
    # ==========================
    logger.info("Loading dataset from /data")
    logger.info(f"Files: {os.listdir('/data')}")

    X = np.load("/data/X.npy")
    Y = np.load("/data/Y.npy")

    logger.info(f"Loaded X={X.shape}, Y={Y.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device → {device}")

    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    # Dataset shapes
    num_samples = X.shape[0]
    input_len = X.shape[1]
    input_dim = X.shape[2]
    pred_len = Y.shape[1]
    output_dim = Y.shape[2]

    logger.info(f"Input len={input_len}, input_dim={input_dim}, pred_len={pred_len}, out_dim={output_dim}")

    # ==========================
    # SPLITTING
    # ==========================
    train_end = int(num_samples * config["train_split"])
    val_end = int(num_samples * config["val_split"])

    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val,   Y_val   = X[train_end:val_end], Y[train_end:val_end]
    X_test,  Y_test  = X[val_end:], Y[val_end:]

    logger.info(f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=config["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=config["batch_size"],
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, Y_test),
        batch_size=config["batch_size"],
        shuffle=False
    )

    # ==========================
    # MODEL DEFINITIONS
    # ==========================

    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden_dim,
                num_layers=layers,
                dropout=dropout if layers > 1 else 0,
                batch_first=True
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
            h_exp = hidden.unsqueeze(1).repeat(1, seq_len, 1)
            e = torch.tanh(self.attn(torch.cat((h_exp, encoder_outputs), dim=2)))
            energy = self.v(e).squeeze(2)
            return torch.softmax(energy, dim=1)

    class Decoder(nn.Module):
        def __init__(self, output_dim, hidden_dim, layers, dropout):
            super().__init__()
            self.attention = Attention(hidden_dim)
            self.lstm = nn.LSTM(
                output_dim + hidden_dim, hidden_dim,
                num_layers=layers,
                dropout=dropout if layers > 1 else 0,
                batch_first=True
            )
            self.fc_out = nn.Linear(hidden_dim, output_dim)

        def forward(self, y_prev, h, c, encoder_outputs):
            attn_weights = self.attention(h[-1], encoder_outputs)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

            lstm_input = torch.cat((y_prev, context), dim=1).unsqueeze(1)
            output, (h, c) = self.lstm(lstm_input, (h, c))

            pred = self.fc_out(output.squeeze(1))
            return pred, h, c

    class Seq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder(input_dim, config["hidden_dim"],
                                   config["num_layers"], config["dropout"])
            self.decoder = Decoder(output_dim, config["hidden_dim"],
                                   config["num_layers"], config["dropout"])
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

    # ==========================
    # MODEL + OPTIM + SCHEDULER
    # ==========================
    model = Seq2Seq().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["scheduler_T_max"]
    )
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if (config["use_amp"] and device == "cuda") else None

    # ==========================
    # TRAIN LOOP
    # ==========================
    logger.info("==== Training Started ====")

    best_val_loss = float("inf")
    best_state = None

    ckpt_dir = "/data/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(config["epochs"]):
        # TRAIN
        model.train()
        train_loss = 0

        for i, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    preds = model(bx, by, teacher_forcing_ratio=config["teacher_forcing_ratio"])
                    loss = criterion(preds, by)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(bx, by, teacher_forcing_ratio=config["teacher_forcing_ratio"])
                loss = criterion(preds, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()

            train_loss += loss.item() * bx.size(0)

            # MODERATE VERBOSITY: print every 50 batches
            if i % config["log_every_n_batches"] == 0:
                logger.info(f"[Epoch {epoch+1}] Batch {i}/{len(train_loader)} - Loss {loss.item():.4f}")

        train_loss /= len(train_loader.dataset)

        # VALIDATION
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx, y=None, teacher_forcing_ratio=0)
                loss = criterion(preds, by)
                val_loss += loss.item() * bx.size(0)

        val_loss /= len(val_loader.dataset)

        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{config['epochs']} "
            f"| Train Loss={train_loss:.5f} | Val Loss={val_loss:.5f}"
        )

        # SAVE BEST
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            torch.save(best_state, "/data/lstm_weather_model_advanced.pth")
            logger.info(f"✓ Best model updated at epoch {epoch+1}")

        # PERIODIC CHECKPOINT
        if (epoch + 1) % config["save_every_n_epochs"] == 0:
            ckpt_path = f"{ckpt_dir}/lstm_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint → {ckpt_path}")

    logger.info("==== Training Finished ====")
    logger.info(f"Best Val Loss: {best_val_loss:.6f}")
