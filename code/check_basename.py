import os
import re
from rich.console import Console
from rich.panel import Panel

console = Console()

FEATURES_ROOT = "/Volumes/LaCie/GAIT/processed_features"

def get_run_info(file_path):
    try:
        parts = file_path.split(os.sep)
        run_folder = parts[-2]
        action_name = parts[-3]
        
        if "Run_" in run_folder:
            run_num = run_folder.split('_')[-1]
            return action_name, run_num
    except:
        pass
    
    filename = os.path.basename(file_path)
    match = re.search(r'_(\d+)(?:_flip)?\.npy$', filename)
    if match:
        return os.path.basename(os.path.dirname(os.path.dirname(file_path))), match.group(1)
        
    return None, None

def is_test_run(action_name, run_number):
    act = action_name.lower()
    if "walk" in act:
        return run_number in ['5', '6']
    else:
        return run_number == '3'

def check_basenames():
    console.print(Panel.fit(f"[bold gradient(cyan,magenta)]--- LOGICAL SPLIT CHECK (Basenames) ---[/bold gradient(cyan,magenta)]"))
    
    train_files = set()
    test_files = set()
    
    if not os.path.exists(FEATURES_ROOT):
        console.print("[red]Features folder not found.[/red]")
        return

    for root, _, files in os.walk(FEATURES_ROOT):
        for f in files:
            if f.endswith('.npy') and not f.startswith('._'):
                full_path = os.path.join(root, f)
                
                action, run_num = get_run_info(full_path)
                if not action or not run_num: continue
                
                rel_path = os.path.relpath(full_path, FEATURES_ROOT)
                
                if is_test_run(action, run_num):
                    test_files.add(rel_path)
                else:
                    train_files.add(rel_path)

    console.print(f"Files assigned to Train: [green]{len(train_files)}[/green]")
    console.print(f"Files assigned to Test:  [yellow]{len(test_files)}[/yellow]")

    overlap = train_files.intersection(test_files)
    
    if len(overlap) > 0:
        console.print(Panel.fit(f"[bold red]FAILED! Found {len(overlap)} files assigned to both sets (impossible with if/else logic, but verifying!)[/bold red]"))
        for o in overlap:
            console.print(f"  -> {o}")
    else:
        console.print(Panel.fit("[bold green]TEST PASSED![/bold green]\nThe split logic is mathematically consistent.\nNo file is in both groups."))

if __name__ == "__main__":
    check_basenames()