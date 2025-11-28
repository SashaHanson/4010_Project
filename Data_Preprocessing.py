import modal

app = modal.App("weather-preprocessing")

image = (
    modal.Image.debian_slim()
    .pip_install("pandas", "numpy", "scikit-learn")
)

volume = modal.Volume.from_name("dataset", create_if_missing=False)

@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={"/data": volume},
)
def run_preprocessing():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import os   # <-- FIX HERE

    print("Files in /data:", os.listdir("/data"))
    df = pd.read_csv("/data/weather.csv", sep=";")

    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%d.%m.%Y %H:%M")
    df = df.sort_values("DateTime").reset_index(drop=True)
    df.interpolate(method="linear", inplace=True)

    df["year"] = df["DateTime"].dt.year
    df["month"] = df["DateTime"].dt.month
    df["day"] = df["DateTime"].dt.day
    df["hour"] = df["DateTime"].dt.hour
    df["dayofweek"] = df["DateTime"].dt.dayofweek
    df["dayofyear"] = df["DateTime"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    df.dropna(inplace=True)

    feature_cols = [
        'Temperature','Relative Humidity','Wind Speed','Wind Direction',
        'Soil Temperature','Soil Moisture',
        'year','month','day','hour','dayofweek','dayofyear',
        'hour_sin','hour_cos','doy_sin','doy_cos'
    ]

    target_cols = [
        'Temperature','Relative Humidity','Wind Speed',
        'Wind Direction','Soil Temperature','Soil Moisture'
    ]

    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(df[feature_cols]),
        columns=feature_cols
    )

    INPUT_WINDOW = 240
    OUTPUT_WINDOW = 168

    def create_windows(data, input_len, output_len, target_cols):
        X, Y = [], []
        total_len = input_len + output_len
        for i in range(len(data) - total_len):
            past = data.iloc[i:i+input_len].values
            future = data.iloc[i+input_len:i+total_len][target_cols].values
            X.append(past)
            Y.append(future)
        return np.array(X), np.array(Y)

    X, Y = create_windows(scaled_data, INPUT_WINDOW, OUTPUT_WINDOW, target_cols)

    np.save("/data/X.npy", X)
    np.save("/data/Y.npy", Y)

    print("Finished preprocessing on GPU!")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
