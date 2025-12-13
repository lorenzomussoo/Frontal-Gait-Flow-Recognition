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
    print("ERROR: Missing gait_processing.py")
    exit()

console = Console()

DATASET_ROOT = "/Volumes/LaCie/GAIT/dataset"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model_multimodal")
PATH_MAIN_MODEL = os.path.join(MODEL_DIR, "multimodal_svm_walk_stairs.joblib") 
PATH_SLOPE_MODEL = os.path.join(MODEL_DIR, "svm_slope.joblib")
VIDEO_CUT_INDEX = 49152

SECURITY_THRESHOLD = 23.6

TEST_SUBJECT = "Alessio"       
TEST_ACTION = "walk"         
TEST_RUN = "6"   

# TEST_SUBJECT = "Jessica"       
# TEST_ACTION = "stairs_up"         
# TEST_RUN = "3"  

# TEST_SUBJECT = "MariaVittoria"       
# TEST_ACTION = "stairs_down"         
# TEST_RUN = "3"  

# TEST_SUBJECT = "Lorenzo"       
# TEST_ACTION = "walk"         
# TEST_RUN = "6"  

# TEST_SUBJECT = "Romeo"       
# TEST_ACTION = "slope_down"         
# TEST_RUN = "3"  

# TEST_SUBJECT = "Luca"       
# TEST_ACTION = "slope_up"         
# TEST_RUN = "3"  

# TEST_SUBJECT = "Laura"       
# TEST_ACTION = "slope_down"         
# TEST_RUN = "3"  

VIDEO_FPS = 180 

IMU_FILES = [
    "Sensor_Free_Acceleration.csv", "Sensor_Orientation_Euler.csv", "Sensor_Magnetic_Field.csv", "Sensor_Orientation_Quat.csv",
    "Segment_Velocity.csv", "Segment_Angular_Velocity.csv", "Segment_Position.csv", "Segment_Orientation_Euler.csv", "Segment_Orientation_Quat.csv",
    "Segment_Acceleration.csv", "Segment_Angular_Acceleration.csv",
    "Joint_Angles_ZXY.csv", "Joint_Angles_XZY.csv", "Ergonomic_Joint_Angles_ZXY.csv", "Ergonomic_Joint_Angles_XZY.csv",
    "Center_of_Mass.csv", "Marker.csv", "Frame_Rate.csv", "TimeStamp.csv"
]

class VisualFilters:
    @staticmethod
    def apply_heatmap(frames):
        if not frames: return []
        heatmap = np.zeros((frames[0].shape[0], frames[0].shape[1]), dtype=np.float32)
        heatmaps = []
        prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev, gray)
            heatmap = cv2.addWeighted(heatmap, 0.95, diff.astype(np.float32), 0.2, 0)
            
            norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            color_map = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            
            blended = cv2.addWeighted(f, 0.6, color_map, 0.4, 0)
            heatmaps.append(blended)
            prev = gray
        return heatmaps

    @staticmethod
    def apply_silhouette(frames):
        if not frames: return []
        out_frames = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            silhouette = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            out_frames.append(silhouette)
        return out_frames

    @staticmethod
    def apply_vibrant_depth(frames):
        if not frames: return []
        out_frames = []
        for f in frames:
            if len(f.shape) == 3:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            else:
                gray = f
            
            norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
            
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            
            final = np.full_like(colored, 255)
            
            final[mask > 0] = colored[mask > 0]
            
            out_frames.append(final)
        return out_frames

    @staticmethod
    def apply_cyber_edges(frames):
        if not frames: return []
        out_frames = []
        for f in frames:
            dark = (f * 0.3).astype(np.uint8)
            
            edges = cv2.Canny(f, 50, 150)
            edges_colored = np.zeros_like(f)
            edges_colored[edges > 0] = (255, 255, 0) # Cyan
            
            glow = cv2.GaussianBlur(edges_colored, (9, 9), 0)
            
            final = cv2.addWeighted(dark, 1.0, edges_colored, 1.0, 0)
            final = cv2.addWeighted(final, 1.0, glow, 0.8, 0)
            
            out_frames.append(final)
        return out_frames

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

def create_dummy_frames(text="NO VIDEO"):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, text, (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return [img] * 60

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
    rgb_path = os.path.join(rgb_dir, os.path.basename(depth_path)) if depth_path and os.path.exists(rgb_dir) else None
    imu_run_dir = os.path.join(subj_dir, 'imu', action, f"Run_{run_num}")

    console.print(f"[yellow]Processing Security:[/yellow] {subject} {action} {run_num}")
    if depth_path:
        vec_depth, _ = extract_video_features_and_images(depth_path)
    else:
        vec_depth = None 

    if vec_depth is None:
        vec_depth = np.zeros(24576, dtype=np.float32)

    vec_rgb = None
    if rgb_path:
        vec_rgb, _ = extract_video_features_and_images(rgb_path)
    
    if vec_rgb is None: 
        vec_rgb = np.zeros_like(vec_depth)

    vec_imu = extract_imu_features(imu_run_dir, {})
    EXPECTED_TOTAL = 54082
    CURRENT_VIDEO_LEN = len(vec_depth) + len(vec_rgb)
    EXPECTED_IMU_LEN = EXPECTED_TOTAL - CURRENT_VIDEO_LEN
    if len(vec_imu) < EXPECTED_IMU_LEN: vec_imu = np.pad(vec_imu, (0, EXPECTED_IMU_LEN - len(vec_imu)))
    elif len(vec_imu) > EXPECTED_IMU_LEN: vec_imu = vec_imu[:EXPECTED_IMU_LEN]

    final_vector = np.concatenate([vec_depth, vec_rgb, vec_imu])
    return final_vector.reshape(1, -1), depth_path, rgb_path

def draw_security_hud(img, state_data, view_names):
    code, label, conf, sub_label, fps, idx = state_data
    H, W = img.shape[:2]
    
    c_green = (0, 255, 0)
    c_yellow = (0, 255, 255)
    c_red = (0, 0, 255)
    c_black = (0, 0, 0)
    c_white = (255, 255, 255)
    c_gray = (150, 150, 150)
    
    if code == 0:
        main_color = c_green
        status_text = "ACCESS GRANTED"
    elif code == 1:
        main_color = c_yellow
        status_text = "SUBJECT UNKNOWN"
    else:
        main_color = c_red
        status_text = "ACCESS DENIED / ERROR"

    cv2.rectangle(img, (0, 0), (W, 100), (20, 20, 20), -1)
    cv2.rectangle(img, (0, 0), (W, 10), main_color, -1) 
    
    cv2.putText(img, status_text, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, main_color, 2)
    
    id_text = f"ID: {label}"
    cv2.putText(img, id_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_white, 2)
    
    info_text = f"CONF: {conf:.1f}%"
    (w_text, h_text), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x_conf = W - w_text - 20
    cv2.putText(img, info_text, (x_conf, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, main_color, 2)
    
    if sub_label:
        font_scale_sub = 0.5
        (w_sub, h_sub), _ = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, 1)
        x_sub = W - w_sub - 20
        cv2.putText(img, sub_label, (x_sub, 85), cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, c_gray, 1)

    cv2.rectangle(img, (0, H-30), (W, H), c_black, -1)
    
    v_left, v_right = view_names
    footer_text = f"CAM-L: {v_left} | CAM-R: {v_right} | FPS: {fps} | FR: {idx} | [TAB] SWITCH | [SPACE] PAUSE | [Q] QUIT"
    
    cv2.putText(img, footer_text, (15, H-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 200, 200), 1)
    
    if (idx // 10) % 2 == 0:
        cv2.circle(img, (W-30, H-15), 6, c_red, -1)

def play_security_loop(view_pairs, security_state):
    if not view_pairs or not view_pairs[0][1]: 
        print("No frames to play")
        return

    total_frames = len(view_pairs[0][1])
    idx = 0
    paused = False
    DELAY = int(1000 / VIDEO_FPS)
    
    cv2.namedWindow("Gait Security System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gait Security System", 1280, 600) 
    
    current_pair_idx = 0
    
    while True:
        name_l, frames_l, name_r, frames_r = view_pairs[current_pair_idx]
        
        img_l = frames_l[idx].copy() if frames_l and idx < len(frames_l) else np.zeros((480, 640, 3), dtype=np.uint8)
        img_r = frames_r[idx].copy() if frames_r and idx < len(frames_r) else np.zeros((480, 640, 3), dtype=np.uint8)
        
        h = 500
        w_l = int(img_l.shape[1] * (h / img_l.shape[0]))
        w_r = int(img_r.shape[1] * (h / img_r.shape[0]))
        
        disp_l = cv2.resize(img_l, (w_l, h))
        disp_r = cv2.resize(img_r, (w_r, h))
        
        display = np.hstack([disp_l, disp_r])

        hud_data = security_state + (VIDEO_FPS, idx)
        draw_security_hud(display, hud_data, (name_l, name_r))

        cv2.imshow("Gait Security System", display)
        
        key = cv2.waitKey(DELAY if not paused else 0)
        if key == ord('q'): break
        elif key == 32: paused = not paused
        elif key == 9: 
            current_pair_idx = (current_pair_idx + 1) % len(view_pairs)
        
        if not paused:
            idx = (idx + 1) % total_frames

    cv2.destroyAllWindows()

def run_security_pro():
    console.print(Panel.fit("[bold gradient(red,yellow)]--- GAIT SECURITY PRO SYSTEM ---[/bold gradient(red,yellow)]"))

    is_slope = "slope" in TEST_ACTION.lower()
    
    if is_slope:
        current_model = PATH_SLOPE_MODEL
        model_type = "SLOPE (IMU Only)"
    else:
        current_model = PATH_MAIN_MODEL
        model_type = "MAIN (Video + IMU)"

    if not os.path.exists(current_model):
        console.print(f"[red]Model not found: {current_model}[/red]")
        return
    
    console.print(f"Model active: [bold cyan]{model_type}[/bold cyan]")
    pipeline = load(current_model)
    classes = pipeline.named_steps['svm'].classes_
    
    X_sample, depth_path, rgb_path = prepare_single_sample(TEST_SUBJECT, TEST_ACTION, TEST_RUN)
    if X_sample is None: return

    if is_slope:
        X_input = X_sample[:, VIDEO_CUT_INDEX:]
    else:
        X_input = X_sample

    proba = pipeline.predict_proba(X_input)[0]
    best_idx = np.argmax(proba)
    p_prob = str(classes[best_idx])
    c_prob = proba[best_idx] * 100
    p_strict = str(pipeline.predict(X_input)[0])
    idx_strict = np.where(classes == p_strict)[0][0]
    c_strict = proba[idx_strict] * 100

    if p_prob != p_strict:
        status_code = 2
        main_label = "SYSTEM CONFLICT"
        conf_display = 0.0
        sub_label = f"Math: {p_strict} ({c_strict:.1f}%) | Prob: {p_prob} ({c_prob:.1f}%)"
        msg = f"[bold red]ACCESS DENIED: Internal Conflict[/bold red]"
    elif c_prob < SECURITY_THRESHOLD:
        status_code = 1
        main_label = "UNKNOWN SUBJECT"
        conf_display = c_prob
        sub_label = f"Best match: {p_prob} (Low Confidence)"
        msg = f"[bold yellow]ACCESS DENIED: Unknown Subject[/bold yellow]"
    else:
        status_code = 0
        main_label = p_prob
        conf_display = c_prob
        sub_label = "Biometric Verification OK"
        msg = f"[bold green]ACCESS GRANTED: Welcome {p_prob} ({c_prob:.1f}%)[/bold green]"

    console.print(Panel.fit(msg, title="Security Decision"))

    frames_d = []
    frames_r = []

    if not is_slope and depth_path and os.path.exists(depth_path):
        console.print(f"[cyan]Loading Video: {os.path.basename(depth_path)}[/cyan]")
        frames_d = load_video_frames_force(depth_path)
        frames_r = load_video_frames_force(rgb_path) if rgb_path else []
    else:
        frames_d = create_dummy_frames(f"NO VIDEO ({model_type})")
        frames_r = []

    view_pairs = []
    
    if frames_r:
        console.print("[dim]Security Scan: Analyzing Movement Patterns...[/dim]")
        
        frames_heatmap = VisualFilters.apply_heatmap(frames_r)
        frames_sil = VisualFilters.apply_silhouette(frames_r)
        frames_vibrant = VisualFilters.apply_vibrant_depth(frames_d)
        frames_cyber = VisualFilters.apply_cyber_edges(frames_r)
        
        frames_trace = []
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
        
        view_pairs.append(("DEPTH MONITOR", frames_d, "TRACE HISTORY", frames_trace))
        
        view_pairs.append(("HEATMAP SCAN", frames_heatmap, "SILHOUETTE", frames_sil))
        
        view_pairs.append(("VIBRANT DEPTH", frames_vibrant, "CYBER EDGES", frames_cyber))

    else:
        view_pairs.append(("NO SIGNAL", frames_d, "NO SIGNAL", frames_d))

    security_state = (status_code, main_label, conf_display, sub_label)
    play_security_loop(view_pairs, security_state)

if __name__ == "__main__":
    run_security_pro()