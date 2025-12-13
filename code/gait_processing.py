import cv2
import numpy as np
import os

IMG_SIZE = (128, 96)
MORPH_KERNEL = np.ones((3, 3), np.uint8)
N_FRAMES_FOR_BG = 20

def build_static_background(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    
    for _ in range(10): cap.read() 

    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(cv2.resize(frame, IMG_SIZE), cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()

    if len(frames) == 0: return None
    return np.median(frames, axis=0).astype(np.uint8)

def create_all_gait_images(video_path, flip_horizontal=False, bg_model=None, invert_color_for_debug=False):

    if bg_model is None: return None, None, None, None, None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None, None, None, None, None

    ret, first_frame = cap.read()
    if not ret: 
        cap.release()
        return None, None, None, None, None, None

    if flip_horizontal: first_frame = cv2.flip(first_frame, 1)

    prev_gray_resized = cv2.cvtColor(cv2.resize(first_frame, IMG_SIZE), cv2.COLOR_BGR2GRAY)
    
    hsv_accumulator = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.float32)
    mask_accumulator = np.zeros((IMG_SIZE[1], IMG_SIZE[0]), dtype=np.float32)
    bw_no_bg_accumulator = np.zeros((IMG_SIZE[1], IMG_SIZE[0]), dtype=np.float32)
    bw_with_bg_accumulator = np.zeros((IMG_SIZE[1], IMG_SIZE[0]), dtype=np.float32) 

    lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=50, qualityLevel=0.1, minDistance=7, blockSize=7)
    
    fg_diff = cv2.absdiff(prev_gray_resized, bg_model)
    _, fg_mask = cv2.threshold(fg_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_mask_clean_lk = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    p0 = cv2.goodFeaturesToTrack(prev_gray_resized, mask=fg_mask_clean_lk, **feature_params)
    
    lk_canvas_color_hybrid = cv2.cvtColor(bg_model, cv2.COLOR_GRAY2BGR)
    lk_canvas_gray_no_bg = np.zeros((IMG_SIZE[1], IMG_SIZE[0]), dtype=np.uint8)
    
    TRACE_COLOR = (0, 255, 0) if not invert_color_for_debug else (255, 0, 255)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if flip_horizontal: frame = cv2.flip(frame, 1)

        frame_resized = cv2.resize(frame, IMG_SIZE)
        gray_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        fg_diff = cv2.absdiff(gray_resized, bg_model)
        _, fg_mask = cv2.threshold(fg_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_CLOSE, MORPH_KERNEL)
        mask_accumulator += (fg_mask_clean > 0).astype(np.float32)

        flow = cv2.calcOpticalFlowFarneback(prev_gray_resized, gray_resized, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        bw_no_bg_frame = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bw_no_bg_frame[fg_mask_clean == 0] = 0 
        bw_no_bg_accumulator += bw_no_bg_frame.astype(np.float32)
        
        hsv_frame = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
        
        angles_deg = ang * 180 / np.pi / 2
        
        if invert_color_for_debug:
            angles_deg = (angles_deg + 90) % 180 
            
        hsv_frame[..., 0] = angles_deg
        hsv_frame[..., 1] = 255
        hsv_frame[..., 2] = bw_no_bg_frame 
        hsv_frame[fg_mask_clean == 0] = 0 
        hsv_accumulator += hsv_frame.astype(np.float32)
        
        if p0 is not None and p0.shape[0] > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray_resized, gray_resized, p0, None, **lk_params)
            if p1 is not None:
                good_new = p1[st==1]
                good_old = p0[st==1]
                h, w = lk_canvas_color_hybrid.shape[:2]
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    if 0 <= int(b) < h and 0 <= int(a) < w and 0 <= int(d) < h and 0 <= int(c) < w:
                        cv2.line(lk_canvas_color_hybrid, (int(a), int(b)), (int(c), int(d)), TRACE_COLOR, 1) 
                        cv2.line(lk_canvas_gray_no_bg, (int(a), int(b)), (int(c), int(d)), 255, 1)
                p0 = good_new.reshape(-1, 1, 2)
            else: p0 = None
        
        if frame_count % 5 == 0 or p0 is None or p0.shape[0] < 10: 
            new_points = cv2.goodFeaturesToTrack(gray_resized, mask=fg_mask_clean, **feature_params)
            if new_points is not None:
                p0 = np.vstack((p0, new_points)) if p0 is not None else new_points

        frame_count += 1
        prev_gray_resized = gray_resized.copy()

    cap.release()
    if frame_count == 0: return None, None, None, None, None, None

    mask_safe = np.maximum(mask_accumulator, 1) 
    mask_safe_bgr = np.maximum(mask_accumulator[..., None], 1)
    
    gofi_color = cv2.cvtColor((hsv_accumulator / mask_safe_bgr).astype(np.uint8), cv2.COLOR_HSV2BGR)
    gofi_mask = cv2.normalize(mask_accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gofi_bw_no_bg = (bw_no_bg_accumulator / mask_safe).astype(np.uint8) 
    gofi_bw_with_bg = cv2.normalize(bw_with_bg_accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return gofi_color, gofi_mask, gofi_bw_no_bg, gofi_bw_with_bg, lk_canvas_gray_no_bg, lk_canvas_color_hybrid