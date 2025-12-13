import os
import shutil
from rich.console import Console
from rich.panel import Panel

console = Console()

DATASET_ROOT = "/Volumes/LaCie/GAIT/dataset" 

def force_clean():
    console.print(Panel.fit(f"[bold red]--- FORCE CLEANUP: {DATASET_ROOT} ---[/bold red]"))

    if not os.path.exists(DATASET_ROOT):
        console.print(f"[red]Error: Dataset not found: {DATASET_ROOT}[/red]")
        return

    export_path = os.path.join(DATASET_ROOT, "Export")
    if os.path.exists(export_path):
        console.print(f"Forcing deletion of: [yellow]Export[/yellow]...")
        shutil.rmtree(export_path, ignore_errors=True)
        console.print("[green]Export folder deleted.[/green]")
    else:
        console.print("[dim]Export folder not found (already cleaned).[/dim]")
    console.print("Reset IMU folders...")
    count = 0
    
    subjects = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
    
    for subject in subjects:
        if subject in ["Export", "System"]: continue
        
        imu_path = os.path.join(DATASET_ROOT, subject, "imu")
        if os.path.exists(imu_path):
            shutil.rmtree(imu_path, ignore_errors=True)
            count += 1
            
    console.print(f"[green]Reset {count} IMU folders.[/green]")
    console.print(Panel.fit("[bold green]CLEANUP COMPLETED. NOW YOU CAN LAUNCH 'organize_imu_fixed.py'[/bold green]"))

if __name__ == "__main__":
    force_clean()