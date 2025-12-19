import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from rich.console import Console
from rich.table import Table
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

def run_analysis(model_path, data_loader_func, title_suffix, save_img_name):
    console.print(Panel.fit(f"[bold yellow]--- ANALYSIS DEEP CONFIDENCE: {title_suffix} ---[/bold yellow]"))

    if not os.path.exists(model_path):
        console.print(f"[red]Model not found: {model_path}[/red]")
        return
    
    pipeline = load(model_path)
    console.print(f"[green]Model loaded: {os.path.basename(model_path)}[/green]")

    data = data_loader_func()
    if data is None: return
    
    if len(data) == 5:
        _, X_test, _, y_test, meta_test = data
    elif len(data) == 4:
        _, X_test, _, y_test = data
        meta_test = ["Unknown_File"] * len(y_test)
    else:
        console.print("[red]Error: data loader format[/red]")
        return
    
    if len(X_test) == 0:
        console.print("[yellow]No test data found.[/yellow]")
        return

    if np.isnan(X_test).any(): X_test = np.nan_to_num(X_test)

    console.print("Computing statistics...")
    
    predictions = pipeline.predict(X_test) 
    
    try:
        all_probs = pipeline.predict_proba(X_test)
        classes = pipeline.named_steps['svm'].classes_
        
        confidence = []
        for i, pred_label in enumerate(predictions):
            class_idx = np.where(classes == pred_label)[0][0]
            conf_val = all_probs[i][class_idx] * 100
            conf_val = all_probs[i][class_idx] * 100
            confidence.append(conf_val)
        
        confidence = np.array(confidence) 
    except AttributeError:
        console.print("[red]The Model does not support predict_proba (probability=False?)[/red]")
        return

    indices_ok = np.where(predictions == y_test)[0]
    indices_err = np.where(predictions != y_test)[0]

    mask_ok = (predictions == y_test)
    conf_ok = confidence[mask_ok]
    conf_err = confidence[~mask_ok]
    table = Table(title=f"Confidence stats: {title_suffix}")
    table.add_column("Metric", style="bold cyan")
    table.add_column("✅ Correct", style="green")
    table.add_column("❌ Errate", style="red")
    
    def get_stats(arr):
        if len(arr) == 0: return "N/A", "N/A", "N/A"
        return f"{np.mean(arr):.2f}%", f"{np.min(arr):.2f}%", f"{np.max(arr):.2f}%"

    mean_ok, min_ok, max_ok = get_stats(conf_ok)
    mean_err, min_err, max_err = get_stats(conf_err)

    table.add_row("Quantity", str(len(conf_ok)), str(len(conf_err)))
    table.add_row("Mean Confidence", mean_ok, mean_err)
    table.add_row("Min Confidence", min_ok, min_err)
    table.add_row("Max Confidence", max_ok, max_err)

    console.print(table)

    console.print(f"\n[bold]--- EXTREME CASES ({title_suffix}) ---[/bold]")

    def print_case(title, idx_list, conf_list, mode="max"):
        if len(idx_list) == 0: return
        
        if mode == "max":
            target_idx_local = np.argmax(conf_list)
        else:
            target_idx_local = np.argmin(conf_list)
            
        real_idx = idx_list[target_idx_local]
        conf_val = conf_list[target_idx_local]
        
        file_info = meta_test[real_idx] 
        if os.sep in file_info: file_info = os.path.basename(file_info)

        real_label = y_test[real_idx]
        pred_label = predictions[real_idx]
        
        color = "green" if real_label == pred_label else "red"
        icon = "🏆" if mode == "max" else "⚠️"
        
        console.print(Panel(
            f"[bold]{file_info}[/bold]\n"
            f"Real: {real_label} -> Predicted: {pred_label}\n"
            f"Confidence: [bold white]{conf_val:.2f}%[/bold white]",
            title=f"{icon} {title}",
            border_style=color
        ))

    print_case("Correct & Most Confident", indices_ok, conf_ok, "max")
    print_case("Correct & Uncertain", indices_ok, conf_ok, "min")
    print_case("Incorrect & Most Confident", indices_err, conf_err, "max")
    print_case("Incorrect & Uncertain", indices_err, conf_err, "min")

    if len(conf_err) > 0:
        suggested_threshold = np.percentile(conf_err, 95) 
        lost_correct = np.sum(conf_ok < suggested_threshold)
        perc_lost = (lost_correct / len(conf_ok)) * 100 if len(conf_ok) > 0 else 0
        
        console.print(Panel.fit(f"""[bold]Strategic Advice ({title_suffix}):[/bold]
[yellow]Suggested Threshold:[/yellow] [bold white]{suggested_threshold:.1f}%[/bold white] (Eliminates 95% errors)
Corrects Lost: {lost_correct} ({perc_lost:.1f}%)""", title="Strategic Advice"))
    else:
        console.print(Panel.fit("[bold green]No errors found! Impossible to calculate optimal threshold.[/bold green]"))

    plt.figure(figsize=(10, 6))
    fixed_bins = np.arange(0, 105, 5)

    if len(conf_ok) > 0:
        sns.histplot(conf_ok, color="green", label="Correct", kde=(len(conf_ok)>1), element="step", alpha=0.3, bins=fixed_bins)
    
    if len(conf_err) > 0:
        sns.histplot(conf_err, color="red", label="Incorrect", kde=(len(conf_err)>1), element="step", alpha=0.3, bins=fixed_bins)
        plt.axvline(x=np.mean(conf_err), color='red', linestyle='--', label='Mean Incorrect')
    
    if len(conf_ok) > 0:
        plt.axvline(x=np.mean(conf_ok), color='green', linestyle='--', label='Mean Correct')
    
    plt.xlabel("Confidence (%)")
    plt.ylabel("Number of Samples")
    plt.legend()
    plt.title(f"Confidence Distribution - {title_suffix}")
    plt.xlim(0, 100)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_img_name)
    console.print(f"[green]Graph saved: {save_img_name}[/green]")

def run_all_analysis():
    run_analysis(PATH_MAIN, load_data, "MAIN (Walk/Stairs)", os.path.join(OUTPUT_DIR, "confidence_analysis_walk_stairs.png"))
    
    print("\n" + "="*60 + "\n")
    
    run_analysis(PATH_SLOPE, load_data_slope, "SLOPE (IMU Only)", os.path.join(OUTPUT_DIR, "confidence_analysis_slope.png"))

if __name__ == "__main__":
    run_all_analysis()