import os
import shutil
from rich.console import Console
from rich.panel import Panel

console = Console()

DATASET_ROOT = "/Volumes/LaCie/GAIT/dataset"

def merge_arthur():
    source = os.path.join(DATASET_ROOT, "Arthur_Kadyrzhanov")
    dest = os.path.join(DATASET_ROOT, "Artur") 

    if not os.path.exists(source):
        console.print("[yellow]Folder 'Arthur_Kadyrzhanov' not found (already merged?)[/yellow]")
        return

    if not os.path.exists(dest):
        console.print(f"[cyan]Renaming {source} -> {dest}[/cyan]")
        os.rename(source, dest)
        return

    console.print(f"[cyan]Manual merge: {source} -> {dest}[/cyan]")
    
    src_imu = os.path.join(source, "imu")
    dst_imu = os.path.join(dest, "imu")
    
    if os.path.exists(src_imu):
        if not os.path.exists(dst_imu):
            shutil.move(src_imu, dst_imu)
            console.print("[green]IMU folder moved successfully.[/green]")
        else:
            for action in os.listdir(src_imu):
                s_act = os.path.join(src_imu, action)
                d_act = os.path.join(dst_imu, action)
                
                if os.path.isdir(s_act):
                    if not os.path.exists(d_act):
                        shutil.move(s_act, d_act)
                    else:
                        for run in os.listdir(s_act):
                            s_run = os.path.join(s_act, run)
                            d_run = os.path.join(d_act, run)
                            
                            if os.path.exists(d_run):
                                new_name = f"{run}_merged"
                                d_run_new = os.path.join(d_act, new_name)
                                shutil.move(s_run, d_run_new)
                                console.print(f"[yellow]Conflict resolved: {run} -> {new_name}[/yellow]")
                            else:
                                shutil.move(s_run, d_run)

    shutil.rmtree(source)
    console.print("[bold green]Arthur merged successfully![/bold green]")

def audit_runs():
    console.print(Panel.fit("[bold]Checking Run Quantity per Subject[/bold]"))
    
    subjects = sorted([d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))])
    
    for subj in subjects:
        imu_path = os.path.join(DATASET_ROOT, subj, "imu")
        if not os.path.exists(imu_path):
            continue
            
        counts = []
        for action in sorted(os.listdir(imu_path)):
            act_path = os.path.join(imu_path, action)
            if os.path.isdir(act_path):
                n_runs = len([x for x in os.listdir(act_path) if x.startswith("Run_")])
                counts.append(f"{action}: {n_runs}")
        
        if counts:
            console.print(f"[bold cyan]{subj}[/bold cyan] -> " + " | ".join(counts))

if __name__ == "__main__":
    merge_arthur()
    print("\n")
    audit_runs()