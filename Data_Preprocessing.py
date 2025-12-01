import modal

# Initialize a Modal application used to encapsulate the computation environment
app = modal.App("weather-preprocessing")

# Define the container image: Debian Slim with required Python packages installed
image = (
    modal.Image.debian_slim()
    .pip_install("pandas", "numpy", "scikit-learn")
)

# Reference an existing Modal volume containing the dataset
volume = modal.Volume.from_name("dataset", create_if_missing=False)

@app.function(
    image=image,
    gpu="A100",             # Allocate an A100 GPU on Modal
    timeout=600,            # Allow up to 10 minutes of execution time
    volumes={"/data": volume},  # Mount the dataset volume inside the container
)
def run_preprocessing():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import os   # Needed for directory listing inside the mounted volume

    # Display all files in the mounted dataset directory to confirm correct volume attachment
    print("Files in /data:", os.listdir("/data"))

    # Read the weather data CSV using semicolon separators )
    df = pd.read_csv("/data/weather.csv", sep=";")

    # Parse datetime strings into pandas datetime objects for easier feature extraction
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%d.%m.%Y %H:%M")

    # Sort chronologically to ensure time-series coherence
    df = df.sort_values("DateTime").reset_index(drop=True)

    # Interpolate missing numerical values using linear interpolation
    df.interpolate(method="linear", inplace=True)

    # Extract calendar-based time features to support ML model performance
    df["year"] = df["DateTime"].dt.year
    df["month"] = df["DateTime"].dt.month
    df["day"] = df["DateTime"].dt.day
    df["hour"] = df["DateTime"].dt.hour
    df["dayofweek"] = df["DateTime"].dt.dayofweek
    df["dayofyear"] = df["DateTime"].dt.dayofyear

    # Encode cyclical time features using sine/cosine transformations. this is needed because time cycles repeat
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    # Remove any remaining NaN values to avoid training issues
    df.dropna(inplace=True)

    # Define input features for model training, including raw weather signals,
    # calendar features, and cyclical encodings
    feature_cols = [
        'Temperature','Relative Humidity','Wind Speed','Wind Direction',
        'Soil Temperature','Soil Moisture',
        'year','month','day','hour','dayofweek','dayofyear',
        'hour_sin','hour_cos','doy_sin','doy_cos'
    ]

    # Specify the target variables predicted by the model
    target_cols = [
        'Temperature','Relative Humidity','Wind Speed',
        'Wind Direction','Soil Temperature','Soil Moisture'
    ]

    # Normalize input features to zero mean and unit variance
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(df[feature_cols]),
        columns=feature_cols
    )

    # Length of the input sequence (past data) and output sequence (future predictions)
    INPUT_WINDOW = 240   # 10 days of hourly measurements
    OUTPUT_WINDOW = 168  # 7 days of hourly predictions

    def create_windows(data, input_len, output_len, target_cols):
        """
        Construct a sliding-window dataset for supervised learning on time-series.
        X contains sequences of past input_len time steps.
        Y contains the corresponding next output_len time steps of selected target variables.
        """
        X, Y = [], []
        total_len = input_len + output_len

        # Loop over all possible windows in the dataset
        for i in range(len(data) - total_len):
            past = data.iloc[i:i+input_len].values
            future = data.iloc[i+input_len:i+total_len][target_cols].values
            X.append(past)
            Y.append(future)

        return np.array(X), np.array(Y)

    # Generate the training dataset using sliding window extraction
    X, Y = create_windows(scaled_data, INPUT_WINDOW, OUTPUT_WINDOW, target_cols)

    # Save preprocessed data back to the dataset volume for later model training
    np.save("/data/X.npy", X)
    np.save("/data/Y.npy", Y)

    print("Finished preprocessing on GPU!")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
