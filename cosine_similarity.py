import logging
import os
import time
import sys
from upload_to_chroma import load_array
import numpy as np

try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.text import Text

# Configure rich console
console = Console()

# Mario-style ASCII art
MARIO_LOGO = """
 █▀▄▀█ ██   █▀▄ █ █▀█   █▀▀ █▀▄▀█ █▄▄ █▀▀ █▀▄ █▀▄ █ █▄ █ █▀▀ █▀
 █ ▀ █ █▄▄  █▄▀ █ █▄█   ██▄ █ ▀ █ █▄█ ██▄ █▄▀ █▄▀ █ █ ▀█ █▄█ ▄█
"""

# Configure logging with rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)

# Display welcome banner
console.print(Panel(Text(MARIO_LOGO, style="green"), border_style="yellow"))
console.print("[yellow]⭐ Starting cosine similarity calculation... [/yellow]")

embedding_file = [
    ('enwiki-00000000-0000-0000-0000', 'enwiki-00005131-0001-0000-0000'),
    ('enwiki-06048186-0015-0000-0006', 'enwiki-06052180-0000-0000-0002')
]

bucket_name = "proposition-vectors"

console.print(f"[cyan]🔍 Loading embeddings from bucket:[/cyan] [green]{bucket_name}[/green]")
embed_list = load_array(bucket_name, embedding_file)
# embed_list is a generator, total count is 1285
total_matrices = 1285
console.print(f"[cyan]📊 Processing[/cyan] [green]{total_matrices}[/green] [cyan]embedding matrices[/cyan]")

next_iter_emb = None
cos_simil_scores = []

# Mario progress animation
blocks = ["🟥", "🟧", "🟨", "🟩", "🟦", "🟪"]
mario_sprite = "👾"  # Mario-like sprite (using space invader as it's more visible)
coin = "🪙"

with Progress(
    SpinnerColumn(spinner_name="dots"),
    TextColumn("[cyan]{task.description}[/cyan]"),
    BarColumn(complete_style="green", finished_style="green"),
    TextColumn("[yellow]{task.percentage:>3.0f}%[/yellow]"),
    TextColumn("{task.completed}/{task.total}"),
    expand=True
) as progress:
    task = progress.add_task("[yellow]Processing matrices...[/yellow]", total=total_matrices)
    
    for i, mat in enumerate(embed_list):
        # Update progress bar
        progress.update(task, advance=1, description=f"[yellow]Processing matrix {i+1}/{total_matrices}[/yellow]")
        
        # Occasional coin animation
        if i % 10 == 0 and i > 0:
            console.print(f"[yellow]{coin} {i} matrices processed! {coin}[/yellow]", highlight=False)
        
        # Process embedding matrices
        if next_iter_emb is None:
            mat1 = mat[:-1]
            mat2 = mat[1:] 
        else: 
            mat1 = np.concatenate((next_iter_emb[np.newaxis,:], mat[:-1]), axis=0)
            mat2 = mat
        next_iter_emb = mat[-1] 

        cos_simil = np.sum(mat1 * mat2, axis=1)
        norm1 = np.linalg.norm(mat1, axis=1)
        norm2 = np.linalg.norm(mat2, axis=1)

        cos_simil = cos_simil/(norm1 * norm2)
        cos_simil_scores.append(cos_simil)
        
        # Small delay to see animation (comment out if performance is critical)
        time.sleep(0.01)

cos_simil_scores = np.concatenate(cos_simil_scores, axis=0)[:-1]
console.print(f"[cyan]🎮 Final cosine similarity scores shape:[/cyan] [green]{cos_simil_scores.shape}[/green]")

# Mario-style progress completion
console.print("\n[green]" + "=" * 60 + "[/green]")
console.print(f"[bold green]{mario_sprite} LEVEL COMPLETE! {mario_sprite}[/bold green]")
console.print("[green]" + "=" * 60 + "[/green]\n")

# Save the cosine similarity scores to a file
output_file = "cos_simil_scores.npy"
np.save(output_file, cos_simil_scores)
console.print(f"[cyan]💾 Saved cosine similarity scores to[/cyan] [green]{os.path.abspath(output_file)}[/green]")
console.print("[bold yellow]Thanks for playing![/bold yellow]")

