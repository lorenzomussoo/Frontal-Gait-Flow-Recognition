import os
import numpy as np
import pandas as pd
import cv2
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

try:
    from gait_processing import create_all_gait_images, build_static_background, N_FRAMES_FOR_BG
except ImportError:
    print("CRITICAL ERROR: 'gait_processing.py' not found.")
    exit()

console = Console()

DATASET_ROOT = "/Volumes/LaCie/GAIT/dataset"
OUTPUT_ROOT = "/Volumes/LaCie/GAIT/processed_features"
SKIP_EXISTING = True

GLOBAL_VIDEO_LEN = 24576 

IMU_FILES = [
    "Sensor_Free_Acceleration.csv", "Sensor_Orientation_Euler.csv", "Sensor_Magnetic_Field.csv", "Sensor_Orientation_Quat.csv",
    "Segment_Velocity.csv", "Segment_Angular_Velocity.csv", "Segment_Position.csv", "Segment_Orientation_Euler.csv", "Segment_Orientation_Quat.csv",
    "Segment_Acceleration.csv", "Segment_Angular_Acceleration.csv",
    "Joint_Angles_ZXY.csv", "Joint_Angles_XZY.csv", "Ergonomic_Joint_Angles_ZXY.csv", "Ergonomic_Joint_Angles_XZY.csv",
    "Center_of_Mass.csv", "Marker.csv", "Frame_Rate.csv", "TimeStamp.csv"
]

CANONICAL_ACTIONS = {
    "walk": "Walk", "Walk": "Walk",
    "stairs_up": "StairsUp", "StairsUp": "StairsUp", 
    "slope_up": "SlopeUp", "SlopeUp": "SlopeUp",
    "stairs_down": "StairsDown", "StairsDown": "StairsDown", 
    "slope_down": "SlopeDown", "SlopeDown": "SlopeDown"
}

def discover_imu_schema():
    console.print("[yellow]Discovering IMU Schema...[/yellow]")
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

def extract_imu_features(run_path, schema, debug_save_path=None):
    all_stats = []
    debug_lines = ["--- IMU STATS REPORT ---", f"Source: {run_path}", "="*30]
    
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
                    
                    if vals.shape[0] == 0:
                        all_stats.extend([0.0] * expected_len)
                        debug_lines.append(f"\n[{filename}] -> ERROR: Empty Data")
                        continue

                    means = np.mean(vals, axis=0)
                    stds = np.std(vals, axis=0)
                    mins = np.min(vals, axis=0)
                    maxs = np.max(vals, axis=0)
                    rms = np.sqrt(np.mean(vals**2, axis=0))
                    
                    all_stats.extend(np.vstack([means, stds, mins, maxs, rms]).T.flatten())
                    
                    debug_lines.append(f"\n[{filename}] (Cols: {vals.shape[1]})")
                    for i in range(len(means)):
                        line = (f"  Col {i}: "
                                f"Mean={means[i]:.4f}, "
                                f"Std={stds[i]:.4f}, "
                                f"Min={mins[i]:.4f}, "
                                f"Max={maxs[i]:.4f}, "
                                f"RMS={rms[i]:.4f}")
                        debug_lines.append(line)
                        
                except Exception as e:
                    all_stats.extend([0.0] * expected_len)
                    debug_lines.append(f"\n[{filename}] -> EXCEPTION: {e}")
            else: 
                all_stats.extend([0.0] * expected_len)
                debug_lines.append(f"\n[{filename}] -> MISSING FILE")

    if debug_save_path:
        os.makedirs(debug_save_path, exist_ok=True)
        try:
            with open(os.path.join(debug_save_path, "imu_stats.txt"), "w") as f:
                f.write("\n".join(debug_lines))
        except: pass

    return np.array(all_stats, dtype=np.float32)

def extract_video_features(video_path, flip=False, save_dir=None, prefix="", color_invert=False):
    
    if not video_path or not os.path.exists(video_path):
        return np.zeros(GLOBAL_VIDEO_LEN, dtype=np.float32)

    try:
        bg_model = build_static_background(video_path, N_FRAMES_FOR_BG)
        if bg_model is None: return None
        if flip: bg_model = cv2.flip(bg_model, 1)

        results = create_all_gait_images(video_path, flip_horizontal=flip, bg_model=bg_model, invert_color_for_debug=color_invert)
        gofi_color, gofi_mask, img_flow, gofi_bg, img_lk, lk_color = results

        if img_flow is None: return None
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            base = f"{prefix}" if not flip else f"{prefix}_flip"
            cv2.imwrite(os.path.join(save_dir, f"{base}_GOFI_Color.png"), gofi_color)
            cv2.imwrite(os.path.join(save_dir, f"{base}_LK_Trace.png"), lk_color)
            if not flip: cv2.imwrite(os.path.join(save_dir, f"{base}_Mask.png"), gofi_mask)

        if img_lk is not None:
            return np.concatenate([img_flow.flatten(), img_lk.flatten()])
        else:
            return img_flow.flatten()

    except Exception:
        return np.zeros(GLOBAL_VIDEO_LEN, dtype=np.float32)

def main():
    console.print(Panel.fit("[bold gradient(cyan,magenta)]--- MULTIMODAL EXTRACTOR ---[/bold gradient(cyan,magenta)]"))
    
    schema = discover_imu_schema()
    total_imu_feats = sum([c * 5 for c in schema.values()])
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    subjects = sorted([d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))])
    
    processed_count = 0
    skipped_count = 0
    
    for subj in track(subjects, description="Elaboration Subjects..."):
        console.print(f"[bold white]Processing Subject: {subj}[/bold white]")
        
        subj_path = os.path.join(DATASET_ROOT, subj)
        depth_root = os.path.join(subj_path, 'depth')
        rgb_root = os.path.join(subj_path, 'rgb') 
        imu_root = os.path.join(subj_path, 'imu')
        
        found_runs = set()

        if os.path.exists(depth_root):
            for action in os.listdir(depth_root):
                if action.startswith('.'): continue
                path = os.path.join(depth_root, action)
                if os.path.isdir(path):
                    canon = CANONICAL_ACTIONS.get(action, action)
                    for f in os.listdir(path):
                        if f.endswith('.avi') and not f.startswith('._'):
                            try:
                                run_n = f.rsplit('_', 1)[1].split('.')[0]
                                found_runs.add((canon, run_n, action)) 
                            except: pass
        
        if os.path.exists(imu_root):
            for action in os.listdir(imu_root):
                if action.startswith('.'): continue
                path = os.path.join(imu_root, action)
                if os.path.isdir(path):
                    canon = CANONICAL_ACTIONS.get(action, action)
                    for d in os.listdir(path):
                        if d.startswith("Run_"):
                            run_n = d.split('_')[1]
                            found_runs.add((canon, run_n, action)) 

        sorted_runs = sorted(list(found_runs), key=lambda x: (x[0] if "Slope" not in x[0] else "z_"+x[0], x[1]))

        for canon_action, run_num, raw_name in sorted_runs:
            
            run_folder_name = f"Run_{run_num}"
            out_run_dir = os.path.join(OUTPUT_ROOT, subj, canon_action, run_folder_name)
            
            filename_base = f"{subj}_{raw_name}_{run_num}"
            npy_path = os.path.join(out_run_dir, f"{filename_base}.npy")
            
            if SKIP_EXISTING and os.path.exists(npy_path):
                skipped_count += 1
                continue
            
            os.makedirs(out_run_dir, exist_ok=True)
            
            debug_main = os.path.join(out_run_dir, "debug")
            os.makedirs(debug_main, exist_ok=True)

            vec_imu = None
            target_imu_folder = None
            if os.path.exists(imu_root):
                for d in os.listdir(imu_root):
                    if CANONICAL_ACTIONS.get(d) == canon_action:
                        target_imu_folder = os.path.join(imu_root, d, run_folder_name)
                        break
            
            if target_imu_folder and os.path.exists(target_imu_folder):
                vec_imu = extract_imu_features(target_imu_folder, schema, debug_save_path=debug_main)
            else:
                with open(os.path.join(debug_main, "imu_stats.txt"), "w") as f:
                    f.write("IMU FOLDER MISSING OR NOT FOUND")
                vec_imu = np.zeros(total_imu_feats, dtype=np.float32)

            full_depth_path = None
            possible_names = [raw_name, canon_action]
            if os.path.exists(depth_root):
                for name in possible_names:
                    p = os.path.join(depth_root, name)
                    if os.path.exists(p):
                        for f in os.listdir(p):
                            if f.endswith(f"_{run_num}.avi") and not f.startswith("._"):
                                full_depth_path = os.path.join(p, f)
                                break
                    if full_depth_path: break
            
            if full_depth_path and os.path.exists(full_depth_path):
                debug_depth = os.path.join(debug_main, "depth")
                debug_rgb = os.path.join(debug_main, "rgb")
                
                vec_depth_norm = extract_video_features(full_depth_path, flip=False, save_dir=debug_depth, prefix="depth")
                vec_depth_flip = extract_video_features(full_depth_path, flip=True, save_dir=os.path.join(debug_depth, "flip"), prefix="depth", color_invert=True)
            else:
                vec_depth_norm = extract_video_features(None)
                vec_depth_flip = extract_video_features(None)

            full_rgb_path = None
            if full_depth_path:
                rgb_vid_dir = os.path.dirname(full_depth_path).replace('depth', 'rgb')
                full_rgb_path = os.path.join(rgb_vid_dir, os.path.basename(full_depth_path))

            if full_rgb_path and os.path.exists(full_rgb_path):
                debug_rgb = os.path.join(debug_main, "rgb")
                vec_rgb_norm = extract_video_features(full_rgb_path, flip=False, save_dir=debug_rgb, prefix="rgb")
                vec_rgb_flip = extract_video_features(full_rgb_path, flip=True, save_dir=os.path.join(debug_rgb, "flip"), prefix="rgb", color_invert=True)
            else:
                vec_rgb_norm = np.zeros(GLOBAL_VIDEO_LEN, dtype=np.float32)
                vec_rgb_flip = np.zeros(GLOBAL_VIDEO_LEN, dtype=np.float32)

            final_norm = np.concatenate([vec_depth_norm, vec_rgb_norm, vec_imu])
            final_flip = np.concatenate([vec_depth_flip, vec_rgb_flip, vec_imu])
            
            np.save(npy_path, final_norm)
            
            flip_name = f"{filename_base}_flip.npy"
            np.save(os.path.join(out_run_dir, flip_name), final_flip)
            
            processed_count += 1
            
            type_msg = "[cyan]VIDEO+IMU[/cyan]" if full_depth_path else "[yellow]IMU ONLY (PAD)[/yellow]"
            print(f"  -> Saved {filename_base} | {type_msg}")

    console.print(Panel.fit(f"[bold green]Processing Completed![/bold green]\nCreated New: {processed_count}\nSkipped (Existing): {skipped_count}"))

if __name__ == "__main__":
    main()