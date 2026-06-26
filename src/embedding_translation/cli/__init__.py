"""etrans — unified CLI for embedding-translation.

Phase 1 ships a minimal `etrans` app with `info` and `list` subcommands. VM's
embedding/dataset/cluster/mapping CLI command surfaces are intentionally not
ported in Phase 1 because most of them are now owned by vectorbench. They
return in a later phase as a thin Typer namespace over the library API.
"""

from __future__ import annotations

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from .. import (
    SUPPORTED_CLUSTERING_METHODS,
    SUPPORTED_MAPPING_METHODS,
    __version__,
)

console = Console()

app = typer.Typer(
    name="etrans",
    help="embedding-translation — translate embeddings between vector spaces",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command("info")
def info() -> None:
    """Show package version and what's wired up."""
    rprint(f"[cyan]embedding-translation[/cyan] version [green]{__version__}[/green]")
    rprint(f"  Mappers   : {', '.join(SUPPORTED_MAPPING_METHODS)}")
    rprint(f"  Clustering: {', '.join(SUPPORTED_CLUSTERING_METHODS)}")
    rprint("  Datasets  : provided by [yellow]vectorbench[/yellow]")
    rprint("  Embeddings: provided by [yellow]vectorbench[/yellow]")


@app.command("list-mappers")
def list_mappers() -> None:
    """List available mapping strategies."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Strategy", style="cyan")
    table.add_column("Status", style="green")
    ported = set(SUPPORTED_MAPPING_METHODS)
    all_planned = [
        "procrustes",
        "cca",
        "simple_linear",
        "linear",
        "nonlinear",
        "gromov_wasserstein",
        "la2m",
        "hmoe",
    ]
    for name in all_planned:
        table.add_row(name, "ported" if name in ported else "Phase 2")
    console.print(table)


if __name__ == "__main__":
    app()
