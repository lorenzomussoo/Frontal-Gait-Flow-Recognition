import os
import zipfile
import shutil
import re
import tempfile
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

console = Console()

SOURCE_IMU_ROOT = "/Volumes/LaCie/GAIT/IMUs"
DEST_DATASET_ROOT = "/Volumes/LaCie/GAIT/dataset"

IGNORE_NAMES = ["Export", "Output", "Session", "Mvn", "System", "Log", "__MACOSX"]

ACTION_MAP = {
    "Slope_up": "slope_up", "Stairs_up": "stairs_up",
    "Slope_down": "slope_down", "Stairs_down": "stairs_down",
    "Walk": "walk"
}

SKIPPED_PEOPLE_LOG = set()

def normalize_name(name):
    return name.replace(" ", "_").strip()

def resolve_target_folder(zip_subject_raw, existing_video_subjects):
    clean_zip_name = normalize_name(zip_subject_raw)
    
    if clean_zip_name in IGNORE_NAMES: return None, "SYSTEM_FOLDER"
    if re.search(r'_\d+$', clean_zip_name): return None, "NUMBER_PATTERN"

    if clean_zip_name.startswith("Alessio"):
        parts = clean_zip_name.split('_')
        if len(parts) > 1 and parts[1].upper().startswith('F'):
            return "Alessio_F", "MATCHED_ALESSIO_F"
        else:
            return "Alessio", "MATCHED_ALESSIO"

    sorted_existing = sorted(list(existing_video_subjects), key=len, reverse=True)
    for vid_subj in sorted_existing:
        if "Alessio" in vid_subj: continue 
        if vid_subj.lower() in clean_zip_name.lower():
            return vid_subj, "MATCHED_EXISTING"

    return clean_zip_name, "NEW_CREATED"

def find_subject_from_path(root_path, temp_dir_root):
    current_path = root_path
    while True:
        dirname = os.path.basename(current_path)
        if not current_path.startswith(temp_dir_root): return None
        if current_path == temp_dir_root: return None 
        
        if dirname not in IGNORE_NAMES and not re.match(r'\d{4}_\d{2}_\d{2}', dirname):
            return dirname
        current_path = os.path.dirname(current_path)
    return None

def process_imu_zip(zip_path, existing_subjects, progress):
    zip_name = os.path.basename(zip_path)
    progress.log(f"[yellow]>> Analyzing ZIP: {zip_name}[/yellow]")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            all_files = zf.namelist()
            csv_files_to_extract = [f for f in all_files if f.endswith('.csv') and not f.startswith('__MACOSX')]
            
            if not csv_files_to_extract:
                progress.log(f"[dim]  No CSV files found in {zip_name}[/dim]")
                return

            with tempfile.TemporaryDirectory() as temp_dir:
                zf.extractall(temp_dir, members=csv_files_to_extract)
                
                processed_subjects_in_zip = set()

                for root, dirs, files in os.walk(temp_dir):
                    folder_name = os.path.basename(root)
                    
                    match_action = re.match(r'([a-zA-Z_]+)-(\d+)', folder_name)
                    
                    if match_action:
                        imu_action_raw = match_action.group(1)
                        run_number = int(match_action.group(2))
                        
                        subject_raw = find_subject_from_path(os.path.dirname(root), temp_dir)
                        if not subject_raw: continue

                        target_subject, status = resolve_target_folder(subject_raw, existing_subjects)

                        if status == "NUMBER_PATTERN":
                            if subject_raw not in SKIPPED_PEOPLE_LOG:
                                progress.log(f"[red]  ! Skip: {subject_raw} (Number)[/red]")
                                SKIPPED_PEOPLE_LOG.add(subject_raw)
                            continue
                        elif status == "SYSTEM_FOLDER" or target_subject is None:
                            continue
                            
                        action_clean = ACTION_MAP.get(imu_action_raw)
                        if not action_clean: continue
                            
                        dest_dir = os.path.join(DEST_DATASET_ROOT, target_subject, "imu", action_clean, f"Run_{run_number}")
                        os.makedirs(dest_dir, exist_ok=True)
                            
                        count = 0
                        for f in files:
                            if f.endswith('.csv'):
                                shutil.copy2(os.path.join(root, f), os.path.join(dest_dir, f))
                                count += 1
                        
                        log_key = f"{target_subject}_{action_clean}"
                        if count > 0 and log_key not in processed_subjects_in_zip:
                            progress.log(f"[green]  + Import: {target_subject}[/green] -> {action_clean}")
                            processed_subjects_in_zip.add(log_key)

    except Exception as e:
        progress.log(f"[bold red]ERROR ZIP {zip_name}: {e}[/bold red]")

def main():
    console.print(Panel.fit("[bold gradient(cyan,magenta)]--- IMU Organizer 7.0 (Fast Extraction) ---[/bold gradient(cyan,magenta)]"))
    
    if not os.path.exists(DEST_DATASET_ROOT):
        console.print("[red]Error Root Dataset[/red]")
        return

    existing_video_subjects = set([d for d in os.listdir(DEST_DATASET_ROOT) if os.path.isdir(os.path.join(DEST_DATASET_ROOT, d))])
    
    zip_files = []
    for f in os.listdir(SOURCE_IMU_ROOT):
        if f.startswith("._"): continue 
        if f.endswith(".zip"):
            zip_files.append(os.path.join(SOURCE_IMU_ROOT, f))
    zip_files.sort()

    console.print(f"Zip to process: [yellow]{len(zip_files)}[/yellow]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn()
    ) as progress:
        task = progress.add_task("[cyan]Processing...", total=len(zip_files))
        
        for zip_file in zip_files:
            process_imu_zip(zip_file, existing_video_subjects, progress)
            progress.update(task, advance=1)

    console.print(Panel.fit("[bold green]Completed![/bold green]"))

    if SKIPPED_PEOPLE_LOG:
        console.print(Panel.fit(f"[bold red]SKIPPED SUBJECTS:[/bold red]\n" + "\n".join(sorted(list(SKIPPED_PEOPLE_LOG)))))

if __name__ == "__main__":
    main()