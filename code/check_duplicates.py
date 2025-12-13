import os
import numpy as np
import hashlib
import re
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

console = Console()

FEATURES_ROOT = "/Volumes/LaCie/GAIT/processed_features"

def get_run_info(file_path):
    try:
        parts = file_path.split(os.sep)
        run_folder = parts[-2] 
        action_name = parts[-3]
        if "Run_" in run_folder:
            return action_name, run_folder.split('_')[-1]
    except: pass
    return None, None

def is_test_run(action_name, run_number):
    act = action_name.lower()
    if "walk" in act: return run_number in ['5', '6']
    else: return run_number == '3'

def md5_of_array(arr):
    return hashlib.md5(arr.tobytes()).hexdigest()

def check_feature_duplicates():
    console.print(Panel.fit(f"[bold gradient(cyan,magenta)]--- CHECK FEATURE DUPLICATES (.npy) ---[/bold gradient(cyan,magenta)]"))

    train_hashes = {}
    duplicates = []
    
    all_files = []
    for root, _, files in os.walk(FEATURES_ROOT):
        for f in files:
            if f.endswith('.npy') and not f.startswith('._'):
                all_files.append(os.path.join(root, f))
    
    console.print(f"Checking {len(all_files)} feature vectors...")

    for fpath in track(all_files, description="Checking vectors..."):
        try:
            vec = np.load(fpath)
            
            if np.all(vec == 0):
                pass

            vec_hash = md5_of_array(vec)
            
            action, run_num = get_run_info(fpath)
            if not action or not run_num: continue

            is_test = is_test_run(action, run_num)
            
            if not is_test:
                if vec_hash not in train_hashes:
                    train_hashes[vec_hash] = []
                train_hashes[vec_hash].append(fpath)
            else:
                if vec_hash in train_hashes:
                    duplicates.append({
                        'test_file': fpath,
                        'train_matches': train_hashes[vec_hash]
                    })
                    
        except Exception as e:
            console.print(f"[red]Error reading {fpath}: {e}[/red]")

    if len(duplicates) > 0:
        console.print(Panel.fit(f"[bold red]ALARM LEAKAGE! Found {len(duplicates)} test vectors IDENTICAL to training[/bold red]"))
        for dup in duplicates[:10]:
            console.print(f"\n[red]Test File:[/red] {os.path.basename(dup['test_file'])}")
            console.print(f"[green]Copies in Train:[/green]")
            for tm in dup['train_matches']:
                console.print(f"  -> {os.path.basename(tm)}")
    else:
        console.print(Panel.fit("[bold green]TEST PASSED! \U0001F389[/bold green]\nNo feature vector of the Test Set exists in the Training Set.\nThe mathematical Data Leakage is excluded."))

if __name__ == "__main__":
    check_feature_duplicates()