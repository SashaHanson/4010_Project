"""
Script to upload the weather dataset CSV file to a Modal volume.
"""
import os
import modal

# Modal token is set via CLI profile (4010project)
# Use: modal profile activate 4010project

# Create Modal app
app = modal.App("weather-data-uploader")

# Create a volume for storing the dataset
volume = modal.Volume.from_name("dataset", create_if_missing=True)

@app.function(
    volumes={"/data": volume},
    files=[modal.File.from_local_file("Hourly Weather Data in Gallipoli (2008-2021).csv")]
)
def upload_dataset():
    """Upload the CSV file to the Modal volume."""
    import shutil
    
    csv_file = "Hourly Weather Data in Gallipoli (2008-2021).csv"
    destination = "/data/weather_data.csv"
    
    # Copy the file to the volume
    shutil.copy(csv_file, destination)
    
    # Commit the volume changes
    volume.commit()
    
    print(f"Successfully uploaded {csv_file} to Modal volume at {destination}")
    
    # Verify the file was uploaded
    if os.path.exists(destination):
        file_size = os.path.getsize(destination)
        print(f"File size: {file_size} bytes")
        return True
    else:
        print("Error: File was not uploaded successfully")
        return False

if __name__ == "__main__":
    with app.run():
        upload_dataset.remote()

