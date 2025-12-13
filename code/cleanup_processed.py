import os
import shutil
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

console = Console()

TARGET_DIR = "/Volumes/LaCie/GAIT/processed_features"

def clean():
    console.print(Panel.fit(f"[bold red]WARNING: YOU ARE ABOUT TO DELETE EVERYTHING IN:[/bold red]\n{TARGET_DIR}"))
    
    if not os.path.exists(TARGET_DIR):
        console.print("[yellow]The folder does not exist. Nothing to clean.[/yellow]")
        return

    if Confirm.ask("Are you sure you want to proceed?"):
        console.print("Deletion in progress...")
        try:
            shutil.rmtree(TARGET_DIR, ignore_errors=True)
            console.print("[bold green]Cleanup completed.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Critical error: {e}[/bold red]")
    else:
        console.print("[dim]Operation cancelled.[/dim]")

if __name__ == "__main__":
    clean()