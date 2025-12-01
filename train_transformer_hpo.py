import modal

app = modal.App("fedformer-hpo")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "matplotlib", "tqdm", "optuna", "plotly", "kaleido")
)

volume = modal.Volume.from_name("dataset")


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 480,  # 8 hours for Optuna optimization
    volumes={"/data": volume},
)
def run_hpo():
    import torch
    import torch.nn as nn
    import numpy as np
    import torch.fft as fft
    import os
    import math
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Files in /data:", os.listdir("/data"))

    # Load dataset
    X = np.load("/data/X.npy")
    Y = np.load("/data/Y.npy")

    # Split into train/validation (90/10) - consistent across all trials
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]

    input_dim = X.shape[2]      # 16 features
    output_dim = Y.shape[2]     # 6 target features
    pred_len = Y.shape[1]       # 168 timesteps
    seq_len = X.shape[1]        # 240 timesteps

    # --------------------------------------------------------------
    # FEDFORMER ARCHITECTURE (Fixed - not optimized)
    # --------------------------------------------------------------

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]

    class FourierBlock(nn.Module):
        def __init__(self, d_model, modes=32, dropout=0.1):
            super().__init__()
            self.modes = modes
            scale = 1 / (d_model * modes)
            self.weights_real = nn.Parameter(scale * torch.randn(modes, d_model))
            self.weights_imag = nn.Parameter(scale * torch.randn(modes, d_model))
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            B, L, C = x.shape
            residual = x
            
            # FFT
            xf = fft.rfft(x, dim=1)
            kept = xf[:, :self.modes, :]

            # Complex multiplication in frequency domain
            real = kept.real * self.weights_real - kept.imag * self.weights_imag
            imag = kept.real * self.weights_imag + kept.imag * self.weights_real
            mixed = real + 1j * imag

            # IFFT
            out = torch.zeros_like(xf)
            out[:, :self.modes, :] = mixed
            x = fft.irfft(out, n=L, dim=1)
            
            # Residual connection and normalization
            x = self.norm(x + residual)
            x = self.dropout(x)
            return x

    class FEDformerEncoder(nn.Module):
        def __init__(self, d_model, n_layers=3, modes=32, dropout=0.1):
            super().__init__()
            self.layers = nn.ModuleList([
                FourierBlock(d_model, modes, dropout) for _ in range(n_layers)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class FEDformerDecoder(nn.Module):
        def __init__(self, d_model, pred_len, modes=32, dropout=0.1):
            super().__init__()
            self.pred_len = pred_len
            self.modes = modes
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(d_model)
            
            # Cross-attention for encoder-decoder interaction
            self.cross_attention = nn.MultiheadAttention(
                d_model, num_heads=8, dropout=dropout, batch_first=True
            )
            
            # Fourier block for decoder
            self.fourier_block = FourierBlock(d_model, modes, dropout)

        def forward(self, decoder_input, encoder_output):
            # Cross-attention: decoder queries attend to encoder keys/values
            attn_out, _ = self.cross_attention(
                decoder_input, encoder_output, encoder_output
            )
            decoder_input = decoder_input + attn_out
            
            # Apply Fourier block
            x = self.fourier_block(decoder_input)
            return x

    class FEDformer(nn.Module):
        def __init__(self, input_dim, output_dim, pred_len, 
                     d_model=256, n_encoder_layers=3, n_decoder_layers=2, 
                     modes=32, dropout=0.1):
            super().__init__()
            self.pred_len = pred_len
            self.d_model = d_model

            # Input embedding
            self.embed = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model)
            
            # Encoder
            self.encoder = FEDformerEncoder(d_model, n_encoder_layers, modes, dropout)
            
            # Decoder
            self.decoder_layers = nn.ModuleList([
                FEDformerDecoder(d_model, pred_len, modes, dropout) 
                for _ in range(n_decoder_layers)
            ])
            
            # Output projection
            self.output_projection = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, output_dim)
            )

        def forward(self, x):
            # x: [batch, seq_len, input_dim]
            batch_size = x.shape[0]
            
            # Encode input sequence
            x = self.embed(x)  # [batch, seq_len, d_model]
            x = self.pos_encoder(x)
            encoder_output = self.encoder(x)  # [batch, seq_len, d_model]
            
            # Create decoder input: use last timestep + positional encoding for future
            last_hidden = encoder_output[:, -1:, :]  # [batch, 1, d_model]
            
            # Create decoder queries with positional encoding for future timesteps
            decoder_pos = self.pos_encoder.pe[:, :self.pred_len, :]
            decoder_input = last_hidden.repeat(1, self.pred_len, 1) + decoder_pos
            
            # Decode to generate future sequence
            for decoder_layer in self.decoder_layers:
                decoder_input = decoder_layer(decoder_input, encoder_output)
            
            # Project to output dimension
            output = self.output_projection(decoder_input)  # [batch, pred_len, output_dim]
            
            return output

    # --------------------------------------------------------------
    # OPTUNA OPTIMIZATION FUNCTION
    # --------------------------------------------------------------

    def train_with_hyperparameters(dropout, learning_rate, weight_decay, batch_size):
        """Train model with given hyperparameters and return best validation loss"""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # Create data loaders with specified batch size
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train_t, Y_train_t),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val_t, Y_val_t),
            batch_size=batch_size,
            shuffle=False
        )

        # Build model with specified dropout (matching train_transformer.py architecture)
        model = FEDformer(
            input_dim=input_dim,
            output_dim=output_dim,
            pred_len=pred_len,
            d_model=256,
            n_encoder_layers=3,
            n_decoder_layers=2,
            modes=32,
            dropout=dropout
        ).to(device)

        # Optimizer with specified hyperparameters
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
        criterion = nn.MSELoss()

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(30):
            # Training phase
            model.train()
            train_loss = 0
            num_batches = 0

            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1

            avg_train_loss = train_loss / num_batches
            scheduler.step()

            # Validation phase
            model.eval()
            val_loss = 0
            val_num_batches = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    pred = model(bx)
                    loss = criterion(pred, by)
                    val_loss += loss.item()
                    val_num_batches += 1

            avg_val_loss = val_loss / val_num_batches if val_num_batches > 0 else float('inf')

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return best_val_loss

    # --------------------------------------------------------------
    # OPTUNA OBJECTIVE FUNCTION
    # --------------------------------------------------------------

    def objective(trial):
        """Optuna objective function that suggests hyperparameters and returns validation loss"""
        # Suggest hyperparameters
        dropout = trial.suggest_float("dropout", 0.05, 0.2, step=0.05)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        
        # Train with suggested hyperparameters
        val_loss = train_with_hyperparameters(dropout, learning_rate, weight_decay, batch_size)
        
        return val_loss

    # --------------------------------------------------------------
    # OPTUNA STUDY SETUP AND EXECUTION
    # --------------------------------------------------------------

    print("\n" + "="*80)
    print("Starting Optuna Hyperparameter Optimization")
    print("="*80)
    print("Hyperparameters to optimize:")
    print("  - dropout: [0.05, 0.1, 0.15, 0.2] (categorical)")
    print("  - learning_rate: [1e-4, 1e-2] (log-uniform)")
    print("  - weight_decay: [1e-5, 1e-3] (log-uniform)")
    print("  - batch_size: [16, 32, 64, 128] (categorical)")
    print("\nUsing Optuna's TPE (Tree-structured Parzen Estimator) sampler")
    print("="*80 + "\n")

    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="fedformer_hpo"
    )

    # Run optimization
    n_trials = 100  # Adjust based on your needs
    print(f"Running {n_trials} trials...\n")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Get best trial
    best_trial = study.best_trial
    best_result = {
        "dropout": best_trial.params["dropout"],
        "learning_rate": best_trial.params["learning_rate"],
        "weight_decay": best_trial.params["weight_decay"],
        "batch_size": best_trial.params["batch_size"],
        "val_loss": best_trial.value
    }

    # --------------------------------------------------------------
    # RESULTS AND ANALYSIS
    # --------------------------------------------------------------

    print("\n" + "="*80)
    print("OPTUNA OPTIMIZATION COMPLETED")
    print("="*80)

    # Best trial results
    print(f"\nBest Trial:")
    print(f"  Trial Number: {best_trial.number}")
    print(f"  Validation Loss: {best_result['val_loss']:.6f}")
    print(f"  Hyperparameters:")
    print(f"    dropout: {best_result['dropout']}")
    print(f"    learning_rate: {best_result['learning_rate']:.6f}")
    print(f"    weight_decay: {best_result['weight_decay']:.6f}")
    print(f"    batch_size: {best_result['batch_size']}")

    # Top 5 trials
    print(f"\nTop 5 Trials:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value)
    for i, trial in enumerate(sorted_trials[:5], 1):
        print(f"  {i}. Trial {trial.number}: Val Loss={trial.value:.6f} | "
              f"dropout={trial.params['dropout']}, lr={trial.params['learning_rate']:.6f}, "
              f"wd={trial.params['weight_decay']:.6f}, bs={trial.params['batch_size']}")

    # Save all Optuna study results
    results_path = "/data/optuna_results.txt"
    with open(results_path, "w") as f:
        f.write("Optuna Optimization Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Trials: {len(study.trials)}\n")
        f.write(f"Best Validation Loss: {best_result['val_loss']:.6f}\n")
        f.write(f"Best Trial Number: {best_trial.number}\n\n")
        f.write("All trials sorted by validation loss (best to worst):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<6} {'Trial':<8} {'Val Loss':<12} {'Dropout':<10} {'LR':<12} {'Weight Decay':<15} {'Batch Size':<12}\n")
        f.write("-"*80 + "\n")
        for i, trial in enumerate(sorted_trials, 1):
            f.write(f"{i:<6} {trial.number:<8} {trial.value:<12.6f} {trial.params['dropout']:<10} "
                   f"{trial.params['learning_rate']:<12.6f} {trial.params['weight_decay']:<15.6f} "
                   f"{trial.params['batch_size']:<12}\n")
    print(f"\nSaved all Optuna results to: {results_path}")

    # Save best hyperparameters to text file
    best_params_path = "/data/best_hyperparameters.txt"
    with open(best_params_path, "w") as f:
        f.write("Best Hyperparameters from Optuna HPO\n")
        f.write("="*50 + "\n\n")
        f.write(f"Best Validation Loss: {best_result['val_loss']:.6f}\n")
        f.write(f"Best Trial Number: {best_trial.number}\n\n")
        f.write("Hyperparameters:\n")
        f.write(f"  dropout: {best_result['dropout']}\n")
        f.write(f"  learning_rate: {best_result['learning_rate']}\n")
        f.write(f"  weight_decay: {best_result['weight_decay']}\n")
        f.write(f"  batch_size: {best_result['batch_size']}\n")
    print(f"Saved best hyperparameters to: {best_params_path}")

    # Generate visualization plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plot_dir = "/data/hpo_plots"
        os.makedirs(plot_dir, exist_ok=True)
        
        # 1. Optimization history (using plotly)
        try:
            fig = plot_optimization_history(study)
            fig.write_image(f"{plot_dir}/optuna_optimization_history.png")
            print(f"Saved optimization history plot to: {plot_dir}/optuna_optimization_history.png")
        except Exception as e:
            print(f"Could not generate plotly optimization history plot: {e}")
        
        # 2. Parameter importances (using plotly)
        try:
            if len(study.trials) > 1:  # Need at least 2 trials for importance
                fig = plot_param_importances(study)
                fig.write_image(f"{plot_dir}/optuna_param_importances.png")
                print(f"Saved parameter importances plot to: {plot_dir}/optuna_param_importances.png")
        except Exception as e:
            print(f"Could not generate parameter importances plot: {e}")
        
        # 3. Custom summary plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Optimization history
        ax1 = axes[0]
        trial_numbers = [t.number for t in sorted_trials]
        val_losses = [t.value for t in sorted_trials]
        ax1.plot(trial_numbers, val_losses, 'o-', alpha=0.6, markersize=3, label='Trial Results')
        ax1.axhline(best_result['val_loss'], color='r', linestyle='--', linewidth=2, 
                   label=f'Best: {best_result["val_loss"]:.6f}')
        ax1.set_xlabel('Trial Number', fontsize=11)
        ax1.set_ylabel('Validation Loss', fontsize=11)
        ax1.set_title('Optuna Optimization History', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Summary text
        ax2 = axes[1]
        ax2.axis('off')
        
        summary_text = f"""Best Hyperparameters Found:

Dropout: {best_result['dropout']}
Learning Rate: {best_result['learning_rate']:.6f}
Weight Decay: {best_result['weight_decay']:.6f}
Batch Size: {best_result['batch_size']}

Best Validation Loss: {best_result['val_loss']:.6f}
Best Trial Number: {best_trial.number}
Total Trials: {len(study.trials)}
"""
        ax2.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/optuna_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved summary plot to: {plot_dir}/optuna_summary.png")
    except ImportError:
        print("Matplotlib/Plotly not available, skipping visualization")
    except Exception as e:
        print(f"Could not generate visualization: {e}")

    # Train final model with best hyperparameters
    print("\n" + "="*80)
    print("Training final model with best hyperparameters...")
    print("="*80)

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Create data loaders with best batch size
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, Y_train_t),
        batch_size=int(best_result["batch_size"]),
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val_t, Y_val_t),
        batch_size=int(best_result["batch_size"]),
        shuffle=False
    )

    # Build model with best dropout (matching train_transformer.py architecture)
    best_model = FEDformer(
        input_dim=input_dim,
        output_dim=output_dim,
        pred_len=pred_len,
        d_model=256,
        n_encoder_layers=3,
        n_decoder_layers=2,
        modes=32,
        dropout=best_result["dropout"]
    ).to(device)

    # Optimizer with best hyperparameters
    optimizer = torch.optim.AdamW(
        best_model.parameters(),
        lr=best_result["learning_rate"],
        weight_decay=best_result["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(30):
        # Training phase
        best_model.train()
        train_loss = 0
        num_batches = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = best_model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(best_model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        scheduler.step()

        # Validation phase
        best_model.eval()
        val_loss = 0
        val_num_batches = 0
        with torch.no_grad():
            for bx, by in val_loader:
                pred = best_model(bx)
                loss = criterion(pred, by)
                val_loss += loss.item()
                val_num_batches += 1

        avg_val_loss = val_loss / val_num_batches if val_num_batches > 0 else float('inf')

        print(f"Epoch {epoch+1}/30: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save best model
    model_path = "/data/fedformer_weather_model_best_hpo.pth"
    torch.save(best_model.state_dict(), model_path)
    print(f"\nSaved best model to: {model_path}")
    print(f"Final validation loss: {best_val_loss:.6f}")
    print(f"Model parameters: {sum(p.numel() for p in best_model.parameters()):,}")

    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("="*80)


@app.local_entrypoint()
def main():
    print("Starting FEDformer hyperparameter optimization...")
    run_hpo.remote()
    print("HPO completed!")

