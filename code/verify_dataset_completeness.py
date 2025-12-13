import os
import re
import zipfile
from collections import defaultdict
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

console = Console()

SOURCE_VIDEO_ROOT = "/Volumes/LaCie/GAIT/FirstRun/RGBD"
SOURCE_IMU_ROOT = "/Volumes/LaCie/GAIT/IMUs"
DEST_DATASET_ROOT = "/Volumes/LaCie/GAIT/dataset" 

IGNORE_NAMES = ["Export", "Output", "Session", "Mvn", "System", "Log", "__MACOSX"]

ACTION_MAP = {
    "Slope_up": "slope_up", "Stairs_up": "stairs_up", "Up": "stairs_up", "SlopeUp": "slope_up", "StairsUp": "stairs_up",
    "Slope_down": "slope_down", "Stairs_down": "stairs_down", "Down": "stairs_down", "SlopeDown": "slope_down", "StairsDown": "stairs_down",
    "Walk": "walk", "walk": "walk"
}

expected_video = defaultdict(lambda: defaultdict(int))
expected_imu = defaultdict(lambda: defaultdict(int))
found_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) 

def normalize_subject_name(name):
    return name.replace(" ", "_").strip()

def scan_source_video():
    console.print("[yellow]1. Scanning Video source (.bag)...[/yellow]")
    for root, dirs, files in os.walk(SOURCE_VIDEO_ROOT):
        for f in files:
            if f.endswith('.bag') and not f.startswith('._'):
                parts = root.split(os.sep)
                if len(parts) < 2: continue
                
                action_raw = parts[-1]
                subject_raw = parts[-2]
                
                subject = normalize_subject_name(subject_raw)
                
                action_std = ACTION_MAP.get(action_raw, action_raw.lower())
                
                expected_video[subject][action_std] += 1

def resolve_imu_subject(name, video_subjects):
    if name in IGNORE_NAMES or re.search(r'_\d+$', name): return None
    if name.startswith("Alessio"):
        return "Alessio_F" if "F" in name.split('_')[-1] else "Alessio"
    
    sorted_existing = sorted(list(video_subjects), key=len, reverse=True)
    for v in sorted_existing:
        if "Alessio" in v: continue
        if v.lower() in name.lower(): return v
    return name

def find_subject_in_path(parts):
    for p in reversed(parts[:-1]):
        if p not in IGNORE_NAMES and not re.match(r'\d{4}_\d{2}_\d{2}', p): return p
    return None

def scan_source_imu():
    console.print("[yellow]2. Scanning IMU source (.zip)...[/yellow]")
    known_video = set(expected_video.keys())
    zip_files = [os.path.join(SOURCE_IMU_ROOT, f) for f in os.listdir(SOURCE_IMU_ROOT) if f.endswith('.zip') and not f.startswith('._')]
    
    for z_path in track(zip_files, description="Reading Zip Indexes..."):
        try:
            with zipfile.ZipFile(z_path, 'r') as zf:
                unique_runs = set()
                for name in zf.namelist():
                    if name.startswith('__MACOSX') or not name.endswith('.csv'): continue
                    parts = name.strip('/').split('/')
                    if len(parts) < 2: continue
                    
                    folder = parts[-2]
                    match = re.match(r'([a-zA-Z_]+)-(\d+)', folder)
                    if match:
                        act_raw, run_num = match.groups()
                        subj_raw = find_subject_in_path(parts[:-1])
                        if not subj_raw: continue
                        tgt_subj = resolve_imu_subject(normalize_subject_name(subj_raw), known_video)
                        if not tgt_subj: continue
                        
                        act_std = ACTION_MAP.get(act_raw, act_raw.lower())
                        
                        key = f"{tgt_subj}|{act_std}|{run_num}"
                        if key not in unique_runs:
                            expected_imu[tgt_subj][act_std] += 1
                            unique_runs.add(key)
        except: pass

def scan_destination():
    console.print("[yellow]3. Scanning Final Dataset...[/yellow]")
    if not os.path.exists(DEST_DATASET_ROOT): return

    for subj in os.listdir(DEST_DATASET_ROOT):
        subj_path = os.path.join(DEST_DATASET_ROOT, subj)
        if not os.path.isdir(subj_path): continue
        
        for mod in ['depth', 'rgb', 'ir']:
            mod_path = os.path.join(subj_path, mod)
            if os.path.exists(mod_path):
                for act in os.listdir(mod_path):
                    if act.startswith('.'): continue
                    
                    act_std = ACTION_MAP.get(act, act) 

                    act_path = os.path.join(mod_path, act)
                    if os.path.isdir(act_path):
                        cnt = len([f for f in os.listdir(act_path) if f.endswith('.avi') and not f.startswith('._')])
                        found_data[subj][act_std][mod] += cnt 
        
        imu_path = os.path.join(subj_path, 'imu')
        if os.path.exists(imu_path):
            for act in os.listdir(imu_path):
                if act.startswith('.'): continue
                
                act_std = ACTION_MAP.get(act, act)

                act_path = os.path.join(imu_path, act)
                if os.path.isdir(act_path):
                    cnt = len([d for d in os.listdir(act_path) if d.startswith('Run_')])
                    found_data[subj][act_std]['imu'] += cnt

def generate_report():
    console.print(Panel.fit("[bold gradient(cyan,magenta)]--- FINAL AUDIT (STRICT SNAKE) ---[/bold gradient(cyan,magenta)]"))
    
    all_subjs = sorted(list(set(expected_video.keys()) | set(found_data.keys())))
    ok_count = 0
    err_count = 0
    
    for subj in all_subjs:
        actions = sorted(list(set(expected_video[subj].keys()) | set(found_data[subj].keys())))
        issues = []
        
        for act in actions:
            exp_v = expected_video[subj][act]
            exp_i = expected_imu[subj][act]
            
            got_d = found_data[subj][act]['depth']
            got_r = found_data[subj][act]['rgb']
            got_i = found_data[subj][act]['ir']
            got_imu = found_data[subj][act]['imu']
            
            if exp_v > 0:
                if got_d != exp_v: issues.append(f"[{act}] Depth: Expected {exp_v}, Found {got_d}")
                if got_r != exp_v: issues.append(f"[{act}] RGB: Expected {exp_v}, Found {got_r}")
                if got_i != exp_v: issues.append(f"[{act}] IR: Expected {exp_v}, Found {got_i}")
            
            if exp_i > 0 and got_imu == 0:
                issues.append(f"[{act}] IMU: Expected {exp_i} run, Found 0")
        
        if issues:
            err_count += 1
            console.print(f"[bold red]❌ {subj}[/bold red]")
            for i in issues: console.print(f"   -> {i}")
        else:
            ok_count += 1
            
    console.print(f"\n[green]Subjects OK: {ok_count}[/green] | [red]Subjects with Errors: {err_count}[/red]")
    if err_count == 0:
        console.print(Panel.fit("[bold green]PERFECT DATASET![/bold green]\nAll folders are standardized and complete."))

if __name__ == "__main__":
    scan_source_video()
    scan_source_imu()
    scan_destination()
    generate_report()