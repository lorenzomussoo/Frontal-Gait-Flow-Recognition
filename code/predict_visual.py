import os
import cv2
import numpy as np
import pandas as pd
from joblib import load
from rich.console import Console
from rich.panel import Panel
import warnings

try:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from gait_processing import create_all_gait_images, build_static_background, N_FRAMES_FOR_BG, IMG_SIZE
except ImportError:
    print("ERROR: gait_processing.py not found")
    exit()

console = Console()

DATASET_ROOT = "/Volumes/LaCie/GAIT/dataset"
FEATURES_ROOT = "/Volumes/LaCie/GAIT/processed_features"
MODEL_DIR = "model_multimodal"
PATH_MAIN_MODEL = os.path.join(MODEL_DIR, "multimodal_svm_walk_stairs.joblib") 
PATH_SLOPE_MODEL = os.path.join(MODEL_DIR, "svm_slope.joblib")
VIDEO_CUT_INDEX = 49152

TEST_SUBJECT = "Alessio"       
TEST_ACTION = "walk"         
TEST_RUN = "6"       

# TEST_SUBJECT = "Jessica"       
# TEST_ACTION = "stairs_up"         
# TEST_RUN = "3"     

# TEST_SUBJECT = "Romeo"       
# TEST_ACTION = "slope_down"         
# TEST_RUN = "3"  

# TEST_SUBJECT = "Chiara"       
# TEST_ACTION = "slope_up"         
# TEST_RUN = "3"  

VIDEO_FPS = 120

IMU_FILES = [
    "Sensor_Free_Acceleration.csv", "Sensor_Orientation_Euler.csv", "Sensor_Magnetic_Field.csv", "Sensor_Orientation_Quat.csv",
    "Segment_Velocity.csv", "Segment_Angular_Velocity.csv", "Segment_Position.csv", "Segment_Orientation_Euler.csv", "Segment_Orientation_Quat.csv",
    "Segment_Acceleration.csv", "Segment_Angular_Acceleration.csv",
    "Joint_Angles_ZXY.csv", "Joint_Angles_XZY.csv", "Ergonomic_Joint_Angles_ZXY.csv", "Ergonomic_Joint_Angles_XZY.csv",
    "Center_of_Mass.csv", "Marker.csv", "Frame_Rate.csv", "TimeStamp.csv"
]

FOLDER_MAP = {
    "walk": "Walk", "Walk": "Walk",
    "stairs_up": "StairsUp", "StairsUp": "StairsUp", 
    "slope_up": "SlopeUp", "SlopeUp": "SlopeUp",
    "stairs_down": "StairsDown", "StairsDown": "StairsDown", 
    "slope_down": "SlopeDown", "SlopeDown": "SlopeDown"
}

def load_video_frames_force(path):
    if not os.path.exists(path): return []
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    return frames

def draw_hud(img, title, idx, total, fps, test_subj, p1, c1, p2, c2):
    H, W = img.shape[:2]
    
    status_color = (0, 255, 0) if p1 == test_subj else (0, 0, 255)
    
    cv2.rectangle(img, (0, 0), (W, 120), (25, 25, 25), -1)
    
    cv2.putText(img, f"ANALYSIS SUBJECT: {test_subj}", (15, 30), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    
    if c1 < c2:
        label_p1 = "SVM(Geo):"
        label_p2 = "Prob:    " 
        
        text_p1 = f"{label_p1} {p1}"
        text_p2 = f"{label_p2} {p2}"
        
    else:
        label_p1 = "PRED:"
        label_p2 = "2nd: "
        
        text_p1 = f"{label_p1} {p1} ({c1:.1f}%)"
        text_p2 = f"{label_p2} {p2} ({c2:.1f}%)"

    cv2.putText(img, text_p1, (15, 70), 
                cv2.FONT_HERSHEY_DUPLEX, 1.1, status_color, 2, cv2.LINE_AA)
    
    cv2.putText(img, text_p2, (15, 100), 
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(img, (0, H-40), (W, H), (0, 0, 0), -1)
    
    type_color = (255, 200, 0) if "RGB" in title else (255, 255, 255)
    cv2.putText(img, f"{title}", (15, H-12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, type_color, 2, cv2.LINE_AA)
    
    info = f"Frame: {idx+1}/{total} | Speed: {fps} FPS | [Q] Next"
    cv2.putText(img, info, (200, H-12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

def play_pro_loop(title, frames_d, frames_r=None, preds=None):
    if not frames_d: 
        return

    idx = 0
    paused = False
    DELAY = int(1000 / VIDEO_FPS)
    
    cv2.namedWindow("Gait Player", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gait Player", 1000, 700) 
    
    total_frames = len(frames_d)
    test_s, p1, c1, p2, c2 = preds

    while True:
        img_d = frames_d[idx].copy()
        
        if frames_r and idx < len(frames_r):
            img_r = frames_r[idx].copy()
            h = 500 
            w_d = int(img_d.shape[1] * (h / img_d.shape[0]))
            w_r = int(img_r.shape[1] * (h / img_r.shape[0]))
            
            disp_d = cv2.resize(img_d, (w_d, h))
            disp_r = cv2.resize(img_r, (w_r, h))
            
            display = np.hstack([disp_d, disp_r])
            
            draw_hud(display, f"{title} (Split View)", idx, total_frames, VIDEO_FPS, test_s, p1, c1, p2, c2)
            
        else:
            display = cv2.resize(img_d, (0,0), fx=2.0, fy=2.0)
            draw_hud(display, title, idx, total_frames, VIDEO_FPS, test_s, p1, c1, p2, c2)

        cv2.imshow("Gait Player", display)
        
        key = cv2.waitKey(DELAY if not paused else 0)
        
        if key == ord('q'): break
        elif key == 32: paused = not paused 
        
        if not paused:
            idx = (idx + 1) % total_frames

    cv2.destroyAllWindows()

def extract_imu_features(run_path, schema):
    all_stats = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for filename in IMU_FILES:
            target_cols = schema.get(filename, 0)
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
                        if target_cols > 0: all_stats.extend([0.0] * (target_cols * 5))
                        continue 
                    means = np.mean(vals, axis=0); stds = np.std(vals, axis=0)
                    mins = np.min(vals, axis=0); maxs = np.max(vals, axis=0)
                    rms = np.sqrt(np.mean(vals**2, axis=0))
                    all_stats.extend(np.vstack([means, stds, mins, maxs, rms]).T.flatten())
                except: 
                    if target_cols > 0: all_stats.extend([0.0] * (target_cols * 5))
            else: 
                if target_cols > 0: all_stats.extend([0.0] * (target_cols * 5))
    return np.array(all_stats, dtype=np.float32)

def extract_video_features_and_images(video_path):
    try:
        bg_model = build_static_background(video_path, N_FRAMES_FOR_BG)
        if bg_model is None: return None, {}
        results = create_all_gait_images(video_path, flip_horizontal=False, bg_model=bg_model)
        gofi_color, gofi_mask, img_flow, img_lk, lk_color = results[0], results[1], results[2], results[4], results[5]
        if img_flow is None: return None, {}
        vector = np.concatenate([img_flow.flatten(), img_lk.flatten()]) if img_lk is not None else img_flow.flatten()
        images = {"GOFI": gofi_color, "MASK": cv2.cvtColor(gofi_mask, cv2.COLOR_GRAY2BGR), "TRACE": lk_color}
        return vector, images
    except: return None, {}

def prepare_single_sample(subject, action, run_num):
    subj_dir = os.path.join(DATASET_ROOT, subject)
    
    depth_dir = os.path.join(subj_dir, 'depth', action)
    depth_path = None
    if os.path.exists(depth_dir):
        for f in os.listdir(depth_dir):
            if f.endswith(f"_{run_num}.avi") and not f.startswith("._"):
                depth_path = os.path.join(depth_dir, f); break
    
    rgb_dir = os.path.join(subj_dir, 'rgb', action)
    rgb_path = None
    if depth_path and os.path.exists(rgb_dir):
        rgb_path = os.path.join(rgb_dir, os.path.basename(depth_path))

    imu_run_dir = os.path.join(subj_dir, 'imu', action, f"Run_{run_num}")
    if not os.path.exists(imu_run_dir):
        parent = os.path.join(subj_dir, 'imu')
        if os.path.exists(parent):
            for d in os.listdir(parent):
                if FOLDER_MAP.get(d) == FOLDER_MAP.get(action):
                    imu_run_dir = os.path.join(parent, d, f"Run_{run_num}")
                    break

    console.print(f"[yellow]Analysis:[/yellow] {subject} {action} {run_num}")
    
    if depth_path:
        vec_depth, imgs_depth = extract_video_features_and_images(depth_path)
    else:
        vec_depth, imgs_depth = None, {}

    if vec_depth is None:
        vec_depth = np.zeros(24576, dtype=np.float32) 
        imgs_depth = {} 

    vec_rgb = None
    imgs_rgb = {}
    if rgb_path:
        vec_rgb, imgs_rgb = extract_video_features_and_images(rgb_path)
    if vec_rgb is None: 
        vec_rgb = np.zeros_like(vec_depth)

    vec_imu = extract_imu_features(imu_run_dir, {})
    
    EXPECTED_TOTAL = 54082
    CURRENT_VIDEO_LEN = len(vec_depth) + len(vec_rgb)
    EXPECTED_IMU_LEN = EXPECTED_TOTAL - CURRENT_VIDEO_LEN
    
    if len(vec_imu) < EXPECTED_IMU_LEN: 
        vec_imu = np.pad(vec_imu, (0, EXPECTED_IMU_LEN - len(vec_imu)))
    elif len(vec_imu) > EXPECTED_IMU_LEN: 
        vec_imu = vec_imu[:EXPECTED_IMU_LEN]

    final_vector = np.concatenate([vec_depth, vec_rgb, vec_imu])
    return final_vector.reshape(1, -1), depth_path, rgb_path, imgs_depth, imgs_rgb

def load_reference_images(subject, action):
    folder_action = FOLDER_MAP.get(action, action)
    base_run_dir = os.path.join(FEATURES_ROOT, subject, folder_action, "Run_1", "debug")
    images = {"Depth": {}, "RGB": {}}
    if os.path.exists(base_run_dir):
        for mod in ["depth", "rgb"]:
            d_dir = os.path.join(base_run_dir, mod)
            k = "Depth" if mod == "depth" else "RGB"
            if os.path.exists(d_dir):
                for f in os.listdir(d_dir):
                    if f.startswith("._"): continue
                    path = os.path.join(d_dir, f)
                    if "GOFI_Color" in f: images[k]["GOFI"] = cv2.imread(path)
                    elif "Mask" in f: images[k]["MASK"] = cv2.imread(path)
                    elif "LK_Trace" in f: images[k]["TRACE"] = cv2.imread(path)
    return images

def create_full_matrix(test_d, test_r, ref_imgs, pred_name):
    W, H = 220, 160 
    def make_pair(img_t, img_r, label_t="TEST", label_r="REF"):
        it = cv2.resize(img_t, (W, H)) if img_t is not None else np.zeros((H, W, 3), np.uint8)
        ir = cv2.resize(img_r, (W, H)) if img_r is not None else np.zeros((H, W, 3), np.uint8)
        cv2.putText(it, label_t, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(ir, label_r, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        return np.vstack([it, ir])

    rows = []
    headers = []
    headers.append(np.zeros((30, 40, 3), dtype=np.uint8)) 
    for t in ["ENERGY (GOFI)", "SILHOUETTE (MASK)", "SKELETON (TRACE)"]:
        h_img = np.zeros((30, W, 3), dtype=np.uint8)
        cv2.putText(h_img, t, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        headers.append(h_img)
        headers.append(np.zeros((30, 10, 3), dtype=np.uint8))
    rows.append(np.hstack(headers[:-1]))

    for mod_name, t_data, r_data in [("DEPTH", test_d, ref_imgs["Depth"]), ("RGB", test_r, ref_imgs["RGB"])]:
        cols = []
        for key in ["GOFI", "MASK", "TRACE"]:
            pair = make_pair(t_data.get(key), r_data.get(key), "TEST", f"REF:{pred_name}")
            cols.append(pair)
            cols.append(np.zeros((pair.shape[0], 10, 3), dtype=np.uint8))
        
        row_content = np.hstack(cols[:-1])
        label_img = np.zeros((row_content.shape[0], 40, 3), dtype=np.uint8)
        for i, c in enumerate(mod_name): cv2.putText(label_img, c, (10, 60+i*35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        rows.append(np.hstack([label_img, row_content]))
        rows.append(np.zeros((15, rows[-1].shape[1], 3), dtype=np.uint8))

    return np.vstack(rows[:-1])

def run_demo():
    console.print(Panel.fit("[bold gradient(green,blue)]--- GAIT VISUALIZER ---[/bold gradient(green,blue)]"))

    is_slope = "slope" in TEST_ACTION.lower()
    if is_slope:
        current_model_path = PATH_SLOPE_MODEL
        model_type = "SLOPE (IMU Only)"
    else:
        current_model_path = PATH_MAIN_MODEL
        model_type = "MAIN (Video + IMU)"

    if not os.path.exists(current_model_path): 
        console.print(f"[red]Error: Model not found: {current_model_path}[/red]")
        return

    console.print(f"Loading Model: [bold cyan]{model_type}[/bold cyan]")
    pipeline = load(current_model_path)
    classes = pipeline.named_steps['svm'].classes_
    
    X_sample, depth_path, rgb_path, imgs_depth, imgs_rgb = prepare_single_sample(TEST_SUBJECT, TEST_ACTION, TEST_RUN)
    if X_sample is None: return

    if is_slope:
        console.print("[dim]Slope Mode: No feature video...[/dim]")
        X_input = X_sample[:, VIDEO_CUT_INDEX:]
    else:
        X_input = X_sample

    proba = pipeline.predict_proba(X_input)[0]
    
    top2_idx = np.argsort(proba)[::-1][:2]
    
    p1_prob = str(classes[top2_idx[0]]) 
    c1_prob = proba[top2_idx[0]] * 100
    
    p2_prob = str(classes[top2_idx[1]]) if len(classes) > 1 else "N/A"
    c2_prob = proba[top2_idx[1]] * 100 if len(classes) > 1 else 0.0

    p1_strict = str(pipeline.predict(X_input)[0])
    idx_strict = np.where(classes == p1_strict)[0][0]
    c_strict = proba[idx_strict] * 100
    if p1_prob != p1_strict:
        p1 = p1_strict
        c1 = c_strict
        
        p2 = p1_prob
        c2 = c1_prob
        
        result_text = (
            f"[bold]Disagreement between Probability and Geometry![/]\n"
            f"\n[bold]Real:[/] {TEST_SUBJECT}\n"
            f"[bold]Probability (Visualizer):[/] {p2} ({c2:.1f}%)\n"
            f"[bold red]Mathematical (logic of Train.py):[/] {p1} ({c1:.1f}%) ❌"
        )

        color = "green" if p1_strict == TEST_SUBJECT else "red"
        console.print(Panel.fit(result_text, style=color))
    else:
        p1 = p1_prob
        c1 = c1_prob
        
        p2 = p2_prob
        c2 = c2_prob

        color = "green" if p1_strict == TEST_SUBJECT else "red"
    
    preds_data = (TEST_SUBJECT, p1, c1, p2, c2)

    result_text = f"Results:\n[bold]Real:[/] {TEST_SUBJECT}\n[bold]Predicted:[/] {p1} ({c1:.1f}%)"
    color = "green" if p1 == TEST_SUBJECT else "red"
    console.print(Panel.fit(result_text, style=color))

    if is_slope:
        console.print("[bold yellow]Slope Mode (IMU Only):[/bold yellow] Video and Matrix visualization disabled (No video available).")
        return 

    console.print("[cyan]Loading Video...[/cyan]")
    frames_d = load_video_frames_force(depth_path)
    frames_r = load_video_frames_force(rgb_path) if rgb_path else []
    
    frames_trace = []
    if frames_r:
        console.print("[dim]Computing RGB traces...[/dim]")
        old_gray = cv2.cvtColor(frames_r[0], cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        mask = np.zeros_like(frames_r[0])
        frames_trace.append(frames_r[0])
        for i in range(1, len(frames_r)):
            frame = frames_r[i].copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if p0 is not None:
                p1_flow, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, winSize=(15,15), maxLevel=2)
                
                if p1_flow is not None: 
                    good_new = p1_flow[st==1] 
                    good_old = p0[st==1]
                    for n, o in zip(good_new, good_old):
                        a,b = n.ravel(); c,d = o.ravel()
                        mask = cv2.line(mask, (int(a),int(b)), (int(c),int(d)), (0,255,0), 2)
                        frame = cv2.circle(frame, (int(a),int(b)), 5, (0,0,255), -1)
                    frame = cv2.add(frame, mask)
                    p0 = good_new.reshape(-1,1,2)
            else: 
                p0 = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            old_gray = gray
            frames_trace.append(frame)

    play_pro_loop("DEPTH VIDEO", frames_d, preds=preds_data)

    if frames_trace:
        play_pro_loop("RGB TRACE", frames_trace, preds=preds_data)
    
    console.print("[yellow]Generating Matrix...[/yellow]")
    ref_imgs = load_reference_images(p1, TEST_ACTION)
    mugshot = create_full_matrix(imgs_depth, imgs_rgb, ref_imgs, p1)
    
    if mugshot.shape[0] > 900:
        scale = 900 / mugshot.shape[0]
        mugshot = cv2.resize(mugshot, (0,0), fx=scale, fy=scale)

    cv2.imshow("Gait DNA Matrix", mugshot)
    console.print("[bold]Press any key to close.[/bold]")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()