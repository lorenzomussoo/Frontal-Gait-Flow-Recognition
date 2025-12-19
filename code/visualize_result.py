import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from joblib import load
from rich.console import Console
from rich.panel import Panel

try:
    from train_walk_stairs import load_data
    from train_slope import load_data_slope
except ImportError as e:
    print(f"ERROR IMPORTS: {e}")
    print("Make sure 'train_walk_stairs.py' and 'train_slope.py' are in the same folder.")
    exit()

console = Console()
MODEL_DIR = "model_multimodal"

PATH_MAIN = os.path.join(MODEL_DIR, "multimodal_svm_walk_stairs.joblib")
PATH_SLOPE = os.path.join(MODEL_DIR, "svm_slope.joblib")

OUTPUT_DIR = os.path.join("analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, title, filename):
    classes = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(12, 10))
    cmap = 'Blues' if "Slope" not in title else 'Purples'
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=classes, yticklabels=classes)
    
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(filename)
    console.print(f"[green]Graph saved: {filename}[/green]")
    plt.show()

def analyze_main_model():
    console.print(Panel.fit("[bold cyan]--- ANALYSIS MAIN MODEL (Walk & Stairs) ---[/bold cyan]"))

    if not os.path.exists(PATH_MAIN):
        console.print(f"[red]Main model not found in {PATH_MAIN}[/red]")
        return

    data = load_data()
    if data is None: return
    _, X_test, _, y_test, _ = data
    
    if np.isnan(X_test).any(): X_test = np.nan_to_num(X_test)

    console.print("Loading model and prediction...")
    pipeline = load(PATH_MAIN)
    y_pred = pipeline.predict(X_test)
    
    plot_confusion_matrix(y_test, y_pred, 
                          "Confusion Matrix: Walk & Stairs (Video+IMU)", 
                          os.path.join(OUTPUT_DIR, "confusion_matrix_walk_stairs.png"))

def analyze_slope_model():
    console.print(Panel.fit("[bold magenta]--- ANALYSIS SLOPE MODEL (IMU Only) ---[/bold magenta]"))

    if not os.path.exists(PATH_SLOPE):
        console.print(f"[red]Slope model not found in {PATH_SLOPE}[/red]")
        return

    data = load_data_slope()
    if data is None: return
    _, X_test, _, y_test, _ = data
    
    if len(X_test) == 0:
        console.print("[yellow]No test data found for Slope.[/yellow]")
        return

    if np.isnan(X_test).any(): X_test = np.nan_to_num(X_test)

    console.print("Loading model and prediction...")
    pipeline = load(PATH_SLOPE)
    y_pred = pipeline.predict(X_test)
    
    plot_confusion_matrix(y_test, y_pred, 
                          "Confusion Matrix: Slope (IMU Only)", 
                          os.path.join(OUTPUT_DIR, "confusion_matrix_slope.png"))

def run_all_visualizations():
    console.print("[bold]Starting visualization of both models...[/bold]\n")
    
    analyze_main_model()
    print("\n" + "="*50 + "\n")
    analyze_slope_model()

if __name__ == "__main__":
    run_all_visualizations()