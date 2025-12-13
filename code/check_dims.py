import os
import numpy as np
import pandas as pd
from rich.console import Console

console = Console()

TEST_FILE = "/Volumes/LaCie/GAIT/processed_features/Vito/StairsDown/Run_1/Vito_stairs_down_1.npy"
DATASET_ROOT = "/Volumes/LaCie/GAIT/dataset"

IMU_FILES = [
    "Sensor_Free_Acceleration.csv", "Sensor_Orientation_Euler.csv", "Sensor_Magnetic_Field.csv", "Sensor_Orientation_Quat.csv",
    "Segment_Velocity.csv", "Segment_Angular_Velocity.csv", "Segment_Position.csv", "Segment_Orientation_Euler.csv", "Segment_Orientation_Quat.csv",
    "Segment_Acceleration.csv", "Segment_Angular_Acceleration.csv",
    "Joint_Angles_ZXY.csv", "Joint_Angles_XZY.csv", "Ergonomic_Joint_Angles_ZXY.csv", "Ergonomic_Joint_Angles_XZY.csv",
    "Center_of_Mass.csv", "Marker.csv", "Frame_Rate.csv", "TimeStamp.csv"
]

def check():
    if not os.path.exists(TEST_FILE):
        console.print(f"[red]Error: The file {TEST_FILE} does not exist. Modify the path in the script.[/red]")
        return

    vector = np.load(TEST_FILE)
    total_len = len(vector)
    console.print(f"[bold]Total Vector Length:[/bold] {total_len}")
    console.print("Computing IMU dimension IMU from original CSV files...")
    imu_len = 0
    
    found = False
    for root, _, files in os.walk(DATASET_ROOT):
        for f in IMU_FILES:
            if f in files:
                try:
                    path = os.path.join(root, f)
                    df = pd.read_csv(path, sep=';', nrows=1)
                    if df.shape[1] < 2: df = pd.read_csv(path, sep=',', nrows=1)
                    cols = [c for c in df.columns if "Frame" not in c and "Time" not in c]
                    feat_count = len(cols) * 5
                    imu_len += feat_count
                except: pass
        if imu_len > 0: 
            found = True
            break
            
    console.print(f"[bold]IMU Length:[/bold] {imu_len}")
    
    video_len = total_len - imu_len
    console.print(f"\n[bold green]RESULT:[/bold green]")
    console.print(f"Total ({total_len}) - IMU ({imu_len}) = [bold cyan]{video_len}[/bold cyan]")
    
    if video_len == 24576:
        console.print("[bold green]✅ CONFIRMED: GLOBAL_VIDEO_LEN = 24576 is CORRECT![/bold green]")
    else:
        console.print(f"[bold red]❌ ATTENTION: The correct value should be {video_len}[/bold red]")

if __name__ == "__main__":
    check()