import modal
import os

#this is what we call the app in modal, it will be shown under this name while
#it is being run
app = modal.App("lstm-advanced-training")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "numpy",
        "tqdm",
    )
)

#what volume are we using for this app
volume = modal.Volume.from_name("dataset")

#this explains that we want to use an A100 GPU and run for at most 1 hour
@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    volumes={"/data": volume}
)
#below are the important imports that are needed to run this code
def train_lstm_advanced():
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    import logging, json

    # ==========================
    # CONFIG, if we choose this for HPO, we will be able to tweak some of these parameters
    # ==========================
    config = {
        "seed": 42, #this is the seed for the random number generator
        "batch_size": 64, 
        "epochs": 40, 
        "learning_rate": 1e-3, #this is the learning rate for the optimizer
        "weight_decay": 1e-5, #this is the weight decay for the optimizer
        "hidden_dim": 256, 
        "num_layers": 2, #this is the number of layers for the LSTM
        "dropout": 0.3, 
        "teacher_forcing_ratio": 0.6, #this is the teacher forcing ratio for the LSTM
        "grad_clip": 1.0, #this is the gradient clip for the optimizer
        "train_split": 0.7,
        "val_split": 0.85, 
        "use_amp": True, #this is the use of mixed precision training
        "scheduler_type": "cosine", #this is the scheduler type for the optimizer
        "scheduler_T_max": 40, #this is the T_max for the scheduler
        "save_every_n_epochs": 2, #this is the save every n epochs for the model
        "log_every_n_batches": 100,  #how much information we want to see in the logs
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
    # LOAD DATA, from the modal volume as we save nothing locally on the device
    # ==========================
    logger.info("Loading dataset from /data")
    logger.info(f"Files: {os.listdir('/data')}")

    X = np.load("/data/X.npy") #load the X data from the modal volume
    Y = np.load("/data/Y.npy") #load the Y data from the modal volume

    logger.info(f"Loaded X={X.shape}, Y={Y.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device → {device}")

    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    #Here we unpack the shapes of our tensors so the rest of the code knows
    #what problem it is solving: num_samples = how many sliding windows we have,
    #input_len / input_dim = length and feature count of the past input window,
    #pred_len / output_dim = length and feature count of the future forecast window.
    num_samples = X.shape[0]
    input_len = X.shape[1]
    input_dim = X.shape[2]
    pred_len = Y.shape[1]
    output_dim = Y.shape[2]

    logger.info(f"Input len={input_len}, input_dim={input_dim}, pred_len={pred_len}, out_dim={output_dim}")

    # ==========================
    #SPLITTING, here we split the data into train, validation and test sets, the splits are defined above
    # ==========================
    train_end = int(num_samples * config["train_split"])
    val_end = int(num_samples * config["val_split"])

    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val,   Y_val   = X[train_end:val_end], Y[train_end:val_end]
    X_test,  Y_test  = X[val_end:], Y[val_end:]

    logger.info(f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Here we wrap the raw tensors into PyTorch DataLoaders so training can happen in mini‑batches:
    # - train_loader shuffles the training windows each epoch to improve generalization,
    # - val_loader and test_loader keep a fixed order (no shuffle) so evaluation is stable and repeatable.
    # All three use the same batch_size from the config.
    #DataLoaders
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
    # MODEL DEFINITIONS (AI PART)
    # ==========================
    # We use an encoder–decoder LSTM with an attention mechanism:
    # - The encoder produces a hidden state for every input hour (a "memory" of the past).
    # - At each forecast step, attention compares the decoder’s current hidden state to all
    #   encoder hidden states and learns a set of weights (how important each past hour is).
    # - These weights are used to build a context vector (weighted sum of encoder states),
    #   so the decoder can focus more on the most relevant parts of the history when
    #   predicting the next hour, instead of treating all past timesteps equally.

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
            """
            x: [batch, input_len, input_dim]
            returns:
              - outputs: all hidden states for each timestep (for attention)
              - (h, c): final hidden and cell states of the LSTM
            """
            outputs, (h, c) = self.lstm(x)
            return outputs, (h, c)

    class Attention(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, hidden, encoder_outputs):
            """
            hidden: last hidden state from decoder (query)       [batch, hidden_dim]
            encoder_outputs: all encoder states (keys/values)   [batch, src_len, hidden_dim]
            returns:
              - attention weights over time steps [batch, src_len]
            """
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
            """
            One decoder step:
              - y_prev: previous prediction (or ground truth during teacher forcing)
              - h, c: current LSTM hidden and cell states
              - encoder_outputs: all encoder states (for attention)
            The decoder:
              1) computes attention over encoder_outputs,
              2) builds a context vector (weighted sum),
              3) feeds [y_prev, context] into an LSTM step,
              4) projects the LSTM output to weather targets.
            """
            attn_weights = self.attention(h[-1], encoder_outputs)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

            lstm_input = torch.cat((y_prev, context), dim=1).unsqueeze(1)
            output, (h, c) = self.lstm(lstm_input, (h, c))

            pred = self.fc_out(output.squeeze(1))
            return pred, h, c

    class Seq2Seq(nn.Module):
        def __init__(self):
            """
            Full sequence‑to‑sequence forecaster.
            Given past 240h, it generates a 168‑step future:
              - At each step t, the decoder predicts hour t,
                then that prediction is fed back in as input
                (this is how autoregressive forecasting works).
            During training we sometimes replace the decoder
            input with the true target (teacher forcing) to
            stabilize and speed up learning.
            """
            super().__init__()
            self.encoder = Encoder(input_dim, config["hidden_dim"],
                                   config["num_layers"], config["dropout"])
            self.decoder = Decoder(output_dim, config["hidden_dim"],
                                   config["num_layers"], config["dropout"])
            self.pred_len = pred_len
            self.output_dim = output_dim

        def forward(self, x, y=None, teacher_forcing_ratio=0.5):
            """
            x: past window          [batch, input_len, input_dim]
            y: future window (GT)   [batch, pred_len, output_dim]  (optional during inference)
            teacher_forcing_ratio: probability of using true target
                                     instead of previous prediction as next input.
            """
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
    # Optimizer + scheduler control how learning happens:
    # - Adam: adaptive learning rate per parameter
    # - CosineAnnealingLR: slowly decays LR in a cosine shape over training
    # - GradScaler / autocast: mixed precision for faster GPU training
    model = Seq2Seq().to(device)
    #The optimizer is the part of the training process that updates the model’s weights so the model learns
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    #The scheduler adjusts the learning rate during training based on some rule.                             
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
        #TRAIN, here we train the model for one epoch
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

            # MODERATE VERBOSITY: print every n batches, defined above
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

        #SAVE BEST, here we save the best model so that we can use it to test the model, and we only want the
        #best since we don't want to clog up the modal volume with too many weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            torch.save(best_state, "/data/lstm_weather_model_advanced.pth")
            logger.info(f"✓ Best model updated at epoch {epoch+1}")

        #PERIODIC CHECKPOINT, here we save the model every n epochs, whic is very important makes it
        #so that we don't waste credits on modal, since sometimes the programming crashes
        if (epoch + 1) % config["save_every_n_epochs"] == 0:
            ckpt_path = f"{ckpt_dir}/lstm_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint → {ckpt_path}")

    logger.info("==== Training Finished ====")
    logger.info(f"Best Val Loss: {best_val_loss:.6f}")
