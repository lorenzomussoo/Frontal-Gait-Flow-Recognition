import os
import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

try:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from gait_processing import create_all_gait_images, build_static_background, N_FRAMES_FOR_BG
except ImportError:
    print("CRITICAL ERROR: 'gait_processing.py' not found.")
    exit()

console = Console()

SOURCE_BAGS_ROOT = "/Volumes/LaCie/GAIT/FirstRun/RGBD"  
DATASET_ROOT = "/Volumes/LaCie/GAIT/dataset"            
OUTPUT_FEATURES_ROOT = "/Volumes/LaCie/GAIT/processed_features" 

TARGETS = ["Alessio", "Alessio F"]

CLIPPING_DISTANCE_METERS = 15.0
DEPTH_SCALE = 0.001

ACTION_MAP = {
    "Walk": "walk", 
    "StairsUp": "stairs_up", "Up": "stairs_up",
    "StairsDown": "stairs_down", "Down": "stairs_down"
}

CANONICAL_ACTIONS = {
    "walk": "Walk", "Walk": "Walk",
    "stairs_up": "StairsUp", "StairsUp": "StairsUp", 
    "slope_up": "SlopeUp", "SlopeUp": "SlopeUp",
    "stairs_down": "StairsDown", "StairsDown": "StairsDown", 
    "slope_down": "SlopeDown", "SlopeDown": "SlopeDown"
}

IMU_FILES = [
    "Sensor_Free_Acceleration.csv", "Sensor_Orientation_Euler.csv", "Sensor_Magnetic_Field.csv", "Sensor_Orientation_Quat.csv",
    "Segment_Velocity.csv", "Segment_Angular_Velocity.csv", "Segment_Position.csv", "Segment_Orientation_Euler.csv", "Segment_Orientation_Quat.csv",
    "Segment_Acceleration.csv", "Segment_Angular_Acceleration.csv",
    "Joint_Angles_ZXY.csv", "Joint_Angles_XZY.csv", "Ergonomic_Joint_Angles_ZXY.csv", "Ergonomic_Joint_Angles_XZY.csv",
    "Center_of_Mass.csv", "Marker.csv", "Frame_Rate.csv", "TimeStamp.csv"
]

def get_bag_info_strict(bag_path):
    parts = bag_path.split(os.sep)
    action_raw = parts[-2]
    subject_raw = parts[-3]
    subject_clean = subject_raw.replace(" ", "_")
    return subject_clean, action_raw, subject_raw

def setup_writer(path, w, h, fps, is_color):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    return cv2.VideoWriter(path, fourcc, fps, (w, h), isColor=is_color)

def convert_bag_safe(bag_path, dest_root_dir, filename_base):
    subj_clean, act_raw, _ = get_bag_info_strict(bag_path)
    act_clean = ACTION_MAP.get(act_raw, act_raw)
    
    base_dest = os.path.join(dest_root_dir, subj_clean)
    path_d = os.path.join(base_dest, "depth", act_clean, filename_base + ".avi")
    path_r = os.path.join(base_dest, "rgb", act_clean, filename_base + ".avi")
    path_i = os.path.join(base_dest, "ir", act_clean, filename_base + ".avi")
    
    os.makedirs(os.path.dirname(path_d), exist_ok=True)
    os.makedirs(os.path.dirname(path_r), exist_ok=True)
    os.makedirs(os.path.dirname(path_i), exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)
    
    try:
        profile = pipeline.start(config)
        profile.get_device().as_playback().set_real_time(False) 
    except: return False, "Err Bag"

    out_d, out_r, out_i = None, None, None
    try:
        sd = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        out_d = setup_writer(path_d, sd.width(), sd.height(), 60, False)
        
        sc = profile.get_stream(rs.stream.color).as_video_stream_profile()
        out_r = setup_writer(path_r, sc.width(), sc.height(), 60, True)
        
        si = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        out_i = setup_writer(path_i, si.width(), si.height(), 60, False)
        
        clip = CLIPPING_DISTANCE_METERS / DEPTH_SCALE
        count = 0
        
        while True:
            s, f = pipeline.try_wait_for_frames(2000) 
            if not s: break
            
            d = np.asanyarray(f.get_depth_frame().get_data())
            mask = np.logical_and(d > 0, d < clip)
            norm = np.zeros_like(d, dtype=np.uint8)
            norm[mask] = ((clip - d[mask]) * 255 / clip).astype(np.uint8)
            out_d.write(norm)
            
            c = np.asanyarray(f.get_color_frame().get_data())
            out_r.write(cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
            
            ir = np.asanyarray(f.get_infrared_frame(1).get_data())
            out_i.write(ir)
            
            count += 1
            
    except: return False, "Err Write"
    finally:
        pipeline.stop()
        if out_d: out_d.release()
        if out_r: out_r.release()
        if out_i: out_i.release()
        
    return True, f"{count} frms"

def discover_imu_schema():
    schema = {}
    for root, _, files in os.walk(DATASET_ROOT):
        for needed_file in IMU_FILES:
            if needed_file not in schema and needed_file in files:
                try:
                    path = os.path.join(root, needed_file)
                    df = pd.read_csv(path, sep=';', nrows=1)
                    if df.shape[1] < 2: df = pd.read_csv(path, sep=',', nrows=1)
                    cols = [c for c in df.columns if "Frame" not in c and "Time" not in c]
                    schema[needed_file] = len(cols)
                except: pass
        if len(schema) == len(IMU_FILES): break
    for f in IMU_FILES:
        if f not in schema: schema[f] = 0
    return schema

def extract_imu_features(run_path, schema):
    all_stats = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for filename in IMU_FILES:
            target_cols = schema.get(filename, 0)
            if target_cols == 0: continue
            expected_len = target_cols * 5
            file_path = os.path.join(run_path, filename)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, sep=';')
                    if df.shape[1] < 2: df = pd.read_csv(file_path, sep=',')
                    cols_to_drop = [c for c in df.columns if "Frame" in c or "Time" in c or "Sample" in c]
                    df = df.drop(columns=cols_to_drop, errors='ignore')
                    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
                    vals = df.values
                    if vals.shape[0] == 0 or vals.shape[1] != target_cols:
                        all_stats.extend([0.0] * expected_len); continue
                    means = np.mean(vals, axis=0); stds = np.std(vals, axis=0)
                    mins = np.min(vals, axis=0); maxs = np.max(vals, axis=0)
                    rms = np.sqrt(np.mean(vals**2, axis=0))
                    all_stats.extend(np.vstack([means, stds, mins, maxs, rms]).T.flatten())
                except: all_stats.extend([0.0] * expected_len)
            else: all_stats.extend([0.0] * expected_len)
    return np.array(all_stats, dtype=np.float32)

def extract_video_features(video_path, flip=False, save_dir=None, prefix="", color_invert=False):
    try:
        bg_model = build_static_background(video_path, N_FRAMES_FOR_BG)
        if bg_model is None: return None
        if flip: bg_model = cv2.flip(bg_model, 1)
        
        results = create_all_gait_images(video_path, flip_horizontal=flip, bg_model=bg_model, invert_color_for_debug=color_invert)
        gofi_color, gofi_mask, img_flow, _, img_lk, lk_color = results
        
        if img_flow is None: return None
        
        if save_dir:
            base = f"{prefix}" if not flip else f"{prefix}_flip"
            cv2.imwrite(os.path.join(save_dir, f"{base}_GOFI_Color.png"), gofi_color)
            cv2.imwrite(os.path.join(save_dir, f"{base}_LK_Trace.png"), lk_color)
            if not flip: cv2.imwrite(os.path.join(save_dir, f"{base}_Mask.png"), gofi_mask)
        
        if img_lk is not None:
            return np.concatenate([img_flow.flatten(), img_lk.flatten()])
        else:
            return img_flow.flatten()
    except: return None

def main():
    console.print(Panel.fit(f"[bold gradient(red,blue)]--- FIX & EXTRACT TARGETS: {TARGETS} ---[/bold gradient(red,blue)]"))

    if not os.path.exists(SOURCE_BAGS_ROOT):
        console.print("[red]Source BAG not found![/red]")
        return

    console.print("\n[bold yellow]PHASE 1: VIDEO REGENERATION (SAFE MODE)[/bold yellow]")
    
    bags = []
    for root, _, files in os.walk(SOURCE_BAGS_ROOT):
        for f in files:
            if f.endswith(".bag") and not f.startswith("._"):
                path_parts = root.split(os.sep)
                match = False
                for t in TARGETS:
                    if t in path_parts:
                        match = True
                        break
                if match:
                    bags.append(os.path.join(root, f))
    
    bags.sort()
    counters = {} 

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn()
    ) as p:
        task_vid = p.add_task("[cyan]Regeneration...", total=len(bags))
        
        for path in bags:
            s_clean, a_raw, _ = get_bag_info_strict(path)
            a_clean = ACTION_MAP.get(a_raw, a_raw)
            
            k = f"{s_clean}_{a_clean}"
            if k not in counters: counters[k] = 0
            counters[k] += 1
            fname = f"{s_clean}_{a_clean}_{counters[k]}"
            
            ok, msg = convert_bag_safe(path, DATASET_ROOT, fname)
            if ok: p.log(f"[green]FIXED: {fname} ({msg})[/green]")
            else: p.log(f"[red]ERR: {fname} ({msg})[/red]")
            p.update(task_vid, advance=1)

    console.print("\n[bold yellow]PHASE 2: FEATURE EXTRACTION[/bold yellow]")
    
    schema = discover_imu_schema()
    total_imu_feats = sum([c * 5 for c in schema.values()])
    
    PROCESSED_TARGETS = [t.replace(" ", "_") for t in TARGETS]
    
    processed_count = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn()
    ) as p:
        task_feat = p.add_task("[magenta]Extracting...", total=None) 

        for subj in PROCESSED_TARGETS:
            subj_path = os.path.join(DATASET_ROOT, subj)
            depth_root = os.path.join(subj_path, 'depth')
            rgb_root = os.path.join(subj_path, 'rgb') 
            imu_root = os.path.join(subj_path, 'imu')
            
            if not os.path.exists(depth_root): continue
            
            for action in os.listdir(depth_root):
                if action.startswith('.'): continue
                
                depth_vid_dir = os.path.join(depth_root, action)
                rgb_vid_dir = os.path.join(rgb_root, action)
                
                imu_action_dir = None
                if os.path.exists(imu_root):
                    for d in os.listdir(imu_root):
                        if CANONICAL_ACTIONS.get(action) == CANONICAL_ACTIONS.get(d):
                            imu_action_dir = os.path.join(imu_root, d)
                            break
                
                if not os.path.isdir(depth_vid_dir): continue
                
                for vid_file in os.listdir(depth_vid_dir):
                    if not vid_file.endswith('.avi') or vid_file.startswith('._'): continue
                    
                    try:
                        run_num = vid_file.rsplit('_', 1)[1].split('.')[0]
                        run_folder_name = f"Run_{run_num}"
                    except: continue
                    
                    out_action_dir = os.path.join(OUTPUT_FEATURES_ROOT, subj, CANONICAL_ACTIONS.get(action, action))
                    out_run_dir = os.path.join(out_action_dir, run_folder_name)
                    
                    debug_depth = os.path.join(out_run_dir, "debug", "depth")
                    debug_rgb = os.path.join(out_run_dir, "debug", "rgb")
                    os.makedirs(os.path.join(debug_depth, "flip"), exist_ok=True)
                    os.makedirs(os.path.join(debug_rgb, "flip"), exist_ok=True)
                    
                    out_name_base = vid_file.replace('.avi', '')

                    vec_imu = None
                    if imu_action_dir:
                        run_path = os.path.join(imu_action_dir, run_folder_name)
                        if os.path.exists(run_path):
                            vec_imu = extract_imu_features(run_path, schema)
                    if vec_imu is None: vec_imu = np.zeros(total_imu_feats, dtype=np.float32)

                    full_d = os.path.join(depth_vid_dir, vid_file)
                    vec_d_norm = extract_video_features(full_d, False, debug_depth, "depth")
                    vec_d_flip = extract_video_features(full_d, True, os.path.join(debug_depth, "flip"), "depth", True)
                    if vec_d_norm is None: continue

                    vec_r_norm, vec_r_flip = None, None
                    if os.path.exists(rgb_vid_dir):
                        full_r = os.path.join(rgb_vid_dir, vid_file)
                        if os.path.exists(full_r):
                            vec_r_norm = extract_video_features(full_r, False, debug_rgb, "rgb")
                            vec_r_flip = extract_video_features(full_r, True, os.path.join(debug_rgb, "flip"), "rgb", True)

                    if vec_r_norm is None:
                        vec_r_norm = np.zeros_like(vec_d_norm)
                        vec_r_flip = np.zeros_like(vec_d_flip)

                    final_norm = np.concatenate([vec_d_norm, vec_r_norm, vec_imu])
                    final_flip = np.concatenate([vec_d_flip, vec_r_flip, vec_imu])
                    
                    np.save(os.path.join(out_run_dir, f"{out_name_base}.npy"), final_norm)
                    np.save(os.path.join(out_run_dir, f"{out_name_base}_flip.npy"), final_flip)
                    
                    processed_count += 1
                    p.update(task_feat, advance=1)
                    
    console.print(Panel.fit(f"[bold green]ALL COMPLETED![/bold green]\nSamples regenerated and extracted: {processed_count} (x2)"))

if __name__ == "__main__":
    main()