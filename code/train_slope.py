import os
import numpy as np
import re
import json
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from joblib import dump
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich import box

np.random.seed(42)

warnings.filterwarnings("ignore")
console = Console()

FEATURES_ROOT = "/Volumes/LaCie/GAIT/processed_features"
MODEL_DIR = "model_multimodal"
MODEL_NAME = "svm_slope.joblib" 
PARAMS_FILE = os.path.join(MODEL_DIR, 'best_params_slope.json')

VIDEO_CUT_INDEX = 49152

PERFORM_GRID_SEARCH = False 

SEED = 42

def is_test_run(run_number):
    return run_number == '3'

def get_run_info(file_path):
    filename = os.path.basename(file_path)
    parent_dir = os.path.basename(os.path.dirname(file_path))
    if "Run_" in parent_dir:
        try: return parent_dir.split('_')[-1]
        except: pass
    match = re.search(r'_(\d+)(?:_flip)?\.npy$', filename)
    if match: return match.group(1)
    return '0'

def load_data_slope():
    console.print(Panel.fit(f"[bold magenta]Loading SLOPE Dataset (IMU Only) from: {FEATURES_ROOT}[/bold magenta]"))
    X_train, y_train = [], []
    X_test, y_test = [], []
    meta_test = [] 

    if not os.path.exists(FEATURES_ROOT):
        console.print("[red]Error: Features folder not found![/red]")
        return None

    subjects = sorted([d for d in os.listdir(FEATURES_ROOT) if os.path.isdir(os.path.join(FEATURES_ROOT, d))])

    # --- ESCLUSIONE SPECIFICA DI UNA PERSONA ---
    # PERSONA_DA_ESCLUDERE = "Alessio" 
    # if PERSONA_DA_ESCLUDERE in subjects:
    #     subjects.remove(PERSONA_DA_ESCLUDERE)
    #     console.print(f"[bold red]ATTENZIONE: Ho rimosso {PERSONA_DA_ESCLUDERE} dal training/test![/bold red]")
    # -------------------------------------------
    
    for subj in track(subjects, description="Loading Slopes..."):
        subj_path = os.path.join(FEATURES_ROOT, subj)
        for action in os.listdir(subj_path):
            if action.startswith('.'): continue
            
            if "slope" not in action.lower(): continue
            
            action_path = os.path.join(subj_path, action)
            if not os.path.isdir(action_path): continue
            
            for root, _, files in os.walk(action_path):
                for f in files:
                    if not f.endswith('.npy') or f.startswith('._'): continue
                    file_path = os.path.join(root, f)
                    try:
                        vector = np.load(file_path)
                        
                        if len(vector) > VIDEO_CUT_INDEX:
                            imu_vector = vector[VIDEO_CUT_INDEX:]
                        else:
                            continue 

                    except: continue
                    
                    run_num = get_run_info(file_path)
                    
                    if is_test_run(run_num):
                        X_test.append(imu_vector)
                        y_test.append(subj)
                        meta_test.append(f)
                    else:
                        X_train.append(imu_vector)
                        y_train.append(subj)

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), meta_test

def print_pretty_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    table = Table(title="\nDetailed Performance Slope (IMU)", box=box.ROUNDED)
    table.add_column("Subject", style="magenta", no_wrap=True)
    table.add_column("Precision", justify="center")
    table.add_column("Recall", justify="center")
    table.add_column("F1-Score", justify="center")
    table.add_column("Support", justify="right")

    special_keys = ['accuracy', 'macro avg', 'weighted avg']
    subjects = [k for k in report.keys() if k not in special_keys]
    
    for subj in sorted(subjects):
        metrics = report[subj]
        p, r, f1, s = metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']
        
        def colorize(val):
            if val >= 0.95: return f"[bold green]{val:.2f}[/bold green]"
            if val >= 0.80: return f"[green]{val:.2f}[/green]"
            if val >= 0.60: return f"[yellow]{val:.2f}[/yellow]"
            return f"[bold red]{val:.2f}[/bold red]"

        table.add_row(subj, colorize(p), colorize(r), colorize(f1), str(s))

    table.add_section()
    acc = report['accuracy']
    table.add_row("[bold]ACCURACY[/bold]", "", "", f"[bold blue]{acc:.2f}[/bold blue]", str(len(y_true)))
    console.print(table)

def train_and_evaluate_slope():
    data = load_data_slope()
    if data is None: return
    X_train, X_test, y_train, y_test, meta_test = data
    
    if len(X_train) == 0:
        console.print("[bold red]No Slope data found! Check extract_features.[/bold red]")
        return

    if np.isnan(X_train).any():
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)
        
    console.print(f"[bold]Training Samples:[/bold] {X_train.shape[0]} | [bold]Features (IMU Only):[/bold] {X_train.shape[1]}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()), 
        ('svm', svm.SVC(probability=True, random_state=SEED))
    ])

    final_model = None
    
    if PERFORM_GRID_SEARCH:
        console.print(Panel.fit("[bold magenta]Starting Grid Search Slope (RBF supported)...[/bold magenta]"))
        param_grid = [
            {'pca__n_components': [0.95, 0.99], 'svm__kernel': ['rbf'], 'svm__C': [10, 100, 1000], 'svm__gamma': ['scale', 'auto']},
            {'pca__n_components': [0.95, 0.99], 'svm__kernel': ['linear'], 'svm__C': [1, 10]}
        ]
        search = GridSearchCV(base_pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
        search.fit(X_train, y_train)
        best_params = search.best_params_
        with open(PARAMS_FILE, 'w') as f: json.dump(best_params, f, indent=4)
        final_model = search.best_estimator_
    else:
        console.print(Panel.fit("[bold blue]Fast Slope training...[/bold blue]"))
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, 'r') as f: loaded_params = json.load(f)
            base_pipeline.set_params(**loaded_params)
        else:
            base_pipeline.set_params(pca__n_components=0.99, svm__kernel='rbf', svm__C=100, svm__gamma='scale') 
            
        base_pipeline.fit(X_train, y_train)
        final_model = base_pipeline

    y_pred = final_model.predict(X_test)
    
    print_pretty_classification_report(y_test, y_pred)
    
    wrong_indices = np.where(y_test != y_pred)[0]
    if len(wrong_indices) > 0:
        table = Table(title=f"Errors made ({len(wrong_indices)})", show_lines=True)
        table.add_column("File .NPY", style="dim") 
        table.add_column("Real", style="green")
        table.add_column("Predicted", style="red")
        for i in wrong_indices:
            table.add_row(meta_test[i], y_test[i], y_pred[i])
        console.print(table)
    else:
        console.print("[bold green]No errors![/bold green]")

    dump(final_model, os.path.join(MODEL_DIR, MODEL_NAME))
    console.print(f"[green]Model saved in: {os.path.join(MODEL_DIR, MODEL_NAME)}[/green]")

if __name__ == "__main__":
    train_and_evaluate_slope()