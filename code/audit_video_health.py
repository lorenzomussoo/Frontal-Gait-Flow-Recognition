import os
import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

console = Console()

DATASET_ROOT = "/Volumes/LaCie/GAIT/dataset"

MIN_FRAMES = 100     
MIN_SIZE_MB = 0.5    

def check_video_health():
    console.print(Panel.fit(f"[bold gradient(cyan,magenta)]--- VIDEO HEALTH AUDIT ---[/bold gradient(cyan,magenta)]\nScanning: {DATASET_ROOT}"))
    
    if not os.path.exists(DATASET_ROOT):
        console.print("[red]Dataset not found![/red]")
        return

    all_avi_files = []
    for root, _, files in os.walk(DATASET_ROOT):
        for f in files:
            if f.endswith('.avi') and not f.startswith('._'):
                all_avi_files.append(os.path.join(root, f))

    console.print(f"Found [yellow]{len(all_avi_files)}[/yellow] videos. Analysis in progress...")

    bad_videos = []
    good_count = 0
    total_size = 0

    for vid_path in track(all_avi_files, description="Checking integrity..."):
        try:
            size_bytes = os.path.getsize(vid_path)
            size_mb = size_bytes / (1024 * 1024)
            total_size += size_mb
            
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                bad_videos.append({
                    "path": vid_path,
                    "reason": "CORRUPT (Cannot Open)",
                    "details": "N/A"
                })
                continue

            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frames / fps if fps > 0 else 0
            
            cap.release()
            
            is_bad = False
            reasons = []
            
            if frames < MIN_FRAMES:
                is_bad = True
                reasons.append(f"LOW FRAMES ({frames})")
            
            if size_mb < MIN_SIZE_MB:
                is_bad = True
                reasons.append(f"SMALL SIZE ({size_mb:.2f} MB)")
            
            if is_bad:
                bad_videos.append({
                    "path": vid_path,
                    "reason": " / ".join(reasons),
                    "details": f"{frames} frms, {size_mb:.2f} MB"
                })
            else:
                good_count += 1
                
        except Exception as e:
            bad_videos.append({
                "path": vid_path,
                "reason": f"ERROR: {str(e)}",
                "details": "Crash"
            })

    console.print("\n")
    
    if bad_videos:
        console.print(Panel.fit(f"[bold red]FOUND {len(bad_videos)} PROBLEMATIC VIDEOS[/bold red]", border_style="red"))
        
        table = Table(show_lines=True)
        table.add_column("Subject/Action", style="cyan")
        table.add_column("File", style="bold")
        table.add_column("Problem", style="red")
        table.add_column("Details", style="yellow")
        
        for item in bad_videos:
            parts = item["path"].split(os.sep)
            try:
                subj = parts[-4]
                modality = parts[-3]
                action = parts[-2]
                filename = parts[-1]
                context = f"{subj} [{modality}/{action}]"
            except:
                context = "Unknown"
                filename = os.path.basename(item["path"])
            
            table.add_row(context, filename, item["reason"], item["details"])
            
        console.print(table)
        console.print(f"\n[bold red]ADVICE:[/bold red] Run [yellow]convert_bags_safe.py[/yellow] to regenerate these specific files.")
    
    else:
        console.print(Panel.fit("[bold green]NO CORRUPT VIDEOS FOUND! \U0001F389[/bold green]\nAll videos pass quality checks."))

    summary = Table(title="Dataset Video Summary")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    
    summary.add_row("Total Videos", str(len(all_avi_files)))
    summary.add_row("Healthy Videos", f"[green]{good_count}[/green]")
    summary.add_row("Problematic Videos", f"[red]{len(bad_videos)}[/red]")
    summary.add_row("Total Space on Disk", f"{total_size/1024:.2f} GB")
    
    console.print(summary)

if __name__ == "__main__":
    check_video_health()