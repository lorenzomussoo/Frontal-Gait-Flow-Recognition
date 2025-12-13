import os
import hashlib
import sys
from collections import defaultdict
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

console = Console()

DATASET_ROOT = "/Volumes/LaCie/GAIT/dataset"

def get_run_info_from_video(filepath):
    try:
        filename = os.path.basename(filepath)
        parts = filepath.split(os.sep)
        action = parts[-2] 
        run_num = filename.rsplit('_', 1)[1].split('.')[0]
        
        return action, run_num
    except:
        return None, None

def is_test_run(action_name, run_number):
    act = action_name.lower()
    if "walk" in act: return run_number in ['5', '6']
    else: return run_number == '3'

def get_file_hash(filepath):
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while True:
                data = f.read(65536) 
                if not data: break
                sha256.update(data)
        return sha256.hexdigest()
    except Exception: return None

def find_duplicate_videos():
    console.print(Panel.fit(f"[bold gradient(cyan,magenta)]--- CHECK VIDEO DUPLICATES (RAW AVI) ---[/bold gradient(cyan,magenta)]"))
    
    if not os.path.exists(DATASET_ROOT):
        console.print("[red]Dataset not found.[/red]")
        return

    train_hashes = defaultdict(list)
    test_hashes = defaultdict(list)
    all_videos = []
    
    for root, _, files in os.walk(DATASET_ROOT):
        for f in files:
            if f.endswith('.avi') and not f.startswith('._'):
                all_videos.append(os.path.join(root, f))

    console.print(f"Scanning [yellow]{len(all_videos)}[/yellow] videos...")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn()
    ) as progress:
        task = progress.add_task("[cyan]Hashing...", total=len(all_videos))
        
        for vid_path in all_videos:
            action, run_num = get_run_info_from_video(vid_path)
            
            if not action or not run_num: 
                progress.update(task, advance=1)
                continue

            file_hash = get_file_hash(vid_path)
            if not file_hash: 
                progress.update(task, advance=1)
                continue
            
            if is_test_run(action, run_num):
                test_hashes[file_hash].append(vid_path)
            else:
                train_hashes[file_hash].append(vid_path)
            
            progress.update(task, advance=1)

    common_hashes = set(train_hashes.keys()).intersection(set(test_hashes.keys()))
    
    if len(common_hashes) > 0:
        console.print(Panel.fit(f"[bold red]ALARM! Found {len(common_hashes)} videos identical between Train and Test[/bold red]"))
        for h in common_hashes:
            console.print(f"\n[red]Hash: {h}[/red]")
            console.print("  [bold]In Train:[/bold]")
            for p in train_hashes[h]: console.print(f"   - {p}")
            console.print("  [bold]In Test:[/bold]")
            for p in test_hashes[h]: console.print(f"   - {p}")
    else:
        console.print(Panel.fit("[bold green]TEST PASSED![/bold green]\nThe original Test videos are physically unique."))

if __name__ == "__main__":
    find_duplicate_videos()