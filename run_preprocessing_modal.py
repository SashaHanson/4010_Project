"""
Modal script to run data preprocessing on the weather dataset.
"""
import modal

# Create Modal app
app = modal.App("weather-preprocessing")

# Reference the existing volume
volume = modal.Volume.from_name("dataset", create_if_missing=True)

@app.function(
    image=modal.Image.debian_slim().pip_install(
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0"
    ),
    volumes={"/data": volume},
    timeout=3600,  # 1 hour timeout
)
def run_preprocessing():
    """Run the data preprocessing pipeline."""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import pickle
    import os
    
    # ===========================
    # 1. Load & Sort Data
    # ===========================
    print("Loading data from Modal volume...")
    df = pd.read_csv("/data/weather_data.csv")
    print(f"Loaded {len(df)} rows")
    
    df['DateTime'] = pd.to_datetime(df['DateTime'], format="%d.%m.%Y %H:%M")
    df = df.sort_values("DateTime").reset_index(drop=True)
    
    # ===========================
    # 2. Handle Missing Values
    # ===========================
    print("Handling missing values...")
    df.interpolate(method='linear', inplace=True)
    
    # ===========================
    # 3. Feature Engineering
    # ===========================
    print("Engineering features...")
    df["year"] = df["DateTime"].dt.year
    df["month"] = df["DateTime"].dt.month
    df["day"] = df["DateTime"].dt.day
    df["hour"] = df["DateTime"].dt.hour
    df["dayofweek"] = df["DateTime"].dt.dayofweek
    df["dayofyear"] = df["DateTime"].dt.dayofyear
    
    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    
    df.dropna(inplace=True)
    print(f"After feature engineering: {len(df)} rows")
    
    # ===========================
    # 4. Select Input & Target Variables
    # ===========================
    
    # Variables used for prediction (input)
    feature_cols = [
        'Temperature',
        'Relative Humidity',
        'Wind Speed',
        'Wind Direction',
        'Soil Temperature',
        'Soil Moisture',
        'year','month','day','hour','dayofweek','dayofyear',
        'hour_sin','hour_cos','doy_sin','doy_cos'
    ]
    
    # Targets we want to forecast (output)
    target_cols = [
        'Temperature',
        'Relative Humidity',
        'Wind Speed',
        'Wind Direction',
        'Soil Temperature',
        'Soil Moisture'
    ]
    
    data = df[feature_cols].copy()
    
    # ===========================
    # 5. Scale Features (Fit ONLY on train later)
    # ===========================
    print("Scaling features...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=feature_cols)
    
    # ===========================
    # 6. Sliding Window Generator
    # ===========================
    
    INPUT_WINDOW = 240   # 10 days
    OUTPUT_WINDOW = 168  # 7 days
    
    def create_windows(data, input_len, output_len, target_cols):
        X, Y = [], []
        total_len = input_len + output_len
        
        for i in range(len(data) - total_len):
            past = data.iloc[i : i + input_len].values
            future = data.iloc[i + input_len : i + total_len][target_cols].values
            
            X.append(past)
            Y.append(future)
        
        return np.array(X), np.array(Y)
    
    print("Creating sliding windows...")
    X, Y = create_windows(scaled_data, INPUT_WINDOW, OUTPUT_WINDOW, target_cols)
    print(f"Created {len(X)} windows")
    
    # ===========================
    # 7. Train/Validation/Test Split (time-series)
    # ===========================
    print("Splitting data...")
    train_size = int(len(X) * 0.7)
    val_size   = int(len(X) * 0.15)
    
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    
    X_val = X[train_size : train_size + val_size]
    Y_val = Y[train_size : train_size + val_size]
    
    X_test = X[train_size + val_size :]
    Y_test = Y[train_size + val_size :]
    
    print(f"Train: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # ===========================
    # 8. Save Preprocessed Data
    # ===========================
    print("Saving preprocessed data to Modal volume...")
    
    # Save arrays
    np.save("/data/X_train.npy", X_train)
    np.save("/data/Y_train.npy", Y_train)
    np.save("/data/X_val.npy", X_val)
    np.save("/data/Y_val.npy", Y_val)
    np.save("/data/X_test.npy", X_test)
    np.save("/data/Y_test.npy", Y_test)
    
    # Save scaler for later use
    with open("/data/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    metadata = {
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "INPUT_WINDOW": INPUT_WINDOW,
        "OUTPUT_WINDOW": OUTPUT_WINDOW,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
    }
    
    with open("/data/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    # Commit volume changes
    volume.commit()
    
    print("✓ Preprocessing complete! Data saved to Modal volume.")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - Y_train: {Y_train.shape}")
    print(f"  - X_val: {X_val.shape}")
    print(f"  - Y_val: {Y_val.shape}")
    print(f"  - X_test: {X_test.shape}")
    print(f"  - Y_test: {Y_test.shape}")
    
    return {
        "status": "success",
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
    }

if __name__ == "__main__":
    with app.run():
        result = run_preprocessing.remote()
        print(result)

