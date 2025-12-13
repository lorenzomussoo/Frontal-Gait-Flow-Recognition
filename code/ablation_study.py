import os
import numpy as np
import re
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
import warnings

warnings.filterwarnings("ignore")
console = Console()

FEATURES_ROOT = "/Volumes/LaCie/GAIT/processed_features"

SPLIT_INDEX = 49152 

SEED = 42
np.random.seed(SEED)

def get_run_info(file_path):
    filename = os.path.basename(file_path)
    parent_dir = os.path.basename(os.path.dirname(file_path))
    if "Run_" in parent_dir:
        try: return parent_dir.split('_')[-1]
        except: pass
    match = re.search(r'_(\d+)(?:_flip)?\.npy$', filename)
    if match: return match.group(1)
    return '0'

def is_test_run_main(action_name, run_number):
    act = action_name.lower()
    if "walk" in act: return run_number in ['5', '6']
    return run_number == '3'

def is_test_run_slope(run_number):
    return run_number == '3'

def load_data_main():
    console.print("[cyan]Loading Data MAIN (Walk & Stairs)...[/cyan]")
    X_train, y_train, X_test, y_test = [], [], [], []

    subjects = sorted([d for d in os.listdir(FEATURES_ROOT) if os.path.isdir(os.path.join(FEATURES_ROOT, d))])
    
    for subj in track(subjects, description="Loading Main..."):
        subj_path = os.path.join(FEATURES_ROOT, subj)
        for action in os.listdir(subj_path):
            if action.startswith('.'): continue
            
            if "slope" in action.lower(): continue
            
            action_path = os.path.join(subj_path, action)
            for root, _, files in os.walk(action_path):
                for f in files:
                    if not f.endswith('.npy') or f.startswith('._'): continue
                    try: vector = np.load(os.path.join(root, f))
                    except: continue
                    
                    run_num = get_run_info(f)
                    
                    if is_test_run_main(action, run_num):
                        X_test.append(vector)
                        y_test.append(subj)
                    else:
                        X_train.append(vector)
                        y_train.append(subj)
    
    if len(X_train) > 0:
        return np.nan_to_num(np.array(X_train)), np.nan_to_num(np.array(X_test)), np.array(y_train), np.array(y_test)
    return None

def load_data_slope():
    console.print("[magenta]Loading Data SLOPE (IMU Only)...[/magenta]")
    X_train, y_train, X_test, y_test = [], [], [], []

    subjects = sorted([d for d in os.listdir(FEATURES_ROOT) if os.path.isdir(os.path.join(FEATURES_ROOT, d))])
    
    for subj in track(subjects, description="Loading Slope..."):
        subj_path = os.path.join(FEATURES_ROOT, subj)
        for action in os.listdir(subj_path):
            if action.startswith('.'): continue
            
            if "slope" not in action.lower(): continue
            
            action_path = os.path.join(subj_path, action)
            for root, _, files in os.walk(action_path):
                for f in files:
                    if not f.endswith('.npy') or f.startswith('._'): continue
                    try: 
                        vector = np.load(os.path.join(root, f))
                        if len(vector) > SPLIT_INDEX:
                            imu_vector = vector[SPLIT_INDEX:]
                        else: continue
                    except: continue
                    
                    run_num = get_run_info(f)
                    
                    if is_test_run_slope(run_num):
                        X_test.append(imu_vector)
                        y_test.append(subj)
                    else:
                        X_train.append(imu_vector)
                        y_train.append(subj)
    
    if len(X_train) > 0:
        return np.nan_to_num(np.array(X_train)), np.nan_to_num(np.array(X_test)), np.array(y_train), np.array(y_test)
    return None

def train_segment(name, X_tr, X_te, y_tr, y_te, color, kernel='linear', C=1):
    console.print(f"  [dim]Training {name}...[/dim]")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.99)), 
        ('svm', svm.SVC(kernel=kernel, C=C, probability=False, random_state=SEED)) 
    ])
    
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    
    return acc

def run_ablation_main():
    console.print(Panel.fit("[bold cyan]ABLATION STUDY: MODEL MAIN (Walk/Stairs)[/bold cyan]"))
    
    data = load_data_main()
    if data is None: return
    X_tr_full, X_te_full, y_tr, y_te = data
    
    X_tr_video = X_tr_full[:, :SPLIT_INDEX]
    X_te_video = X_te_full[:, :SPLIT_INDEX]
    
    X_tr_imu = X_tr_full[:, SPLIT_INDEX:]
    X_te_imu = X_te_full[:, SPLIT_INDEX:]
    
    acc_video = train_segment("Only Video", X_tr_video, X_te_video, y_tr, y_te, "cyan", kernel='linear', C=1)
    
    acc_imu = train_segment("Only IMU", X_tr_imu, X_te_imu, y_tr, y_te, "magenta", kernel='linear', C=1)
    
    acc_full = train_segment("Multimodal", X_tr_full, X_te_full, y_tr, y_te, "green", kernel='linear', C=1)
    
    table = Table(title="Results Main Model", show_lines=True)
    table.add_column("Modality", style="bold")
    table.add_column("Accuracy", justify="center")
    table.add_column("Impact Fusion", justify="center")
    
    table.add_row("Only Video", f"{acc_video*100:.2f}%", f"{(acc_video-acc_full)*100:+.2f}%")
    table.add_row("Only IMU", f"{acc_imu*100:.2f}%", f"{(acc_imu-acc_full)*100:+.2f}%")
    table.add_row("[bold green]Fusion[/bold green]", f"[bold green]{acc_full*100:.2f}%[/bold green]", "-")
    
    console.print(table)
    
    best_single = max(acc_video, acc_imu)
    if acc_full > best_single:
        msg = f"The fusion improves by [bold green]+{(acc_full-best_single)*100:.2f}%[/bold green] compared to the best single."
        color = "green"
    elif acc_full == best_single:
        msg = "The fusion equals the best single."
        color = "yellow"
    else:
        msg = f"The fusion worsens by [bold red]{(acc_full-best_single)*100:.2f}%[/bold red] (Feature Dilution)."
        color = "red"
    console.print(Panel(msg, style=color))
    print("\n")

def run_ablation_slope():
    console.print(Panel.fit("[bold magenta]ABLATION STUDY: MODEL SLOPE (Specialist)[/bold magenta]"))
    
    data = load_data_slope()
    if data is None: return
    X_tr, X_te, y_tr, y_te = data
    
    acc_linear = train_segment("Slope (Linear)", X_tr, X_te, y_tr, y_te, "magenta", kernel='linear', C=1)
    acc_rbf = train_segment("Slope (RBF)", X_tr, X_te, y_tr, y_te, "magenta", kernel='rbf', C=100) # Parametri tipici ottimizzati
    
    table = Table(title="Results Slope Model (IMU Only)", show_lines=True)
    table.add_column("Kernel SVM", style="bold")
    table.add_column("Accuracy", justify="center")
    
    table.add_row("Linear", f"{acc_linear*100:.2f}%")
    table.add_row("RBF", f"{acc_rbf*100:.2f}%")
    
    console.print(table)
    console.print(Panel("We only test IMU because video is not available for slopes.\nWe compare Linear vs RBF to justify the kernel choice.", style="magenta"))

if __name__ == "__main__":
    console.print("[bold]STARTING ABLATION STUDY[/bold]\n")
    run_ablation_main()
    run_ablation_slope()