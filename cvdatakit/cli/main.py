"""CLI entry point: ``cvdatakit``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


# ── root group ────────────────────────────────────────────────────────────────

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="cvdatakit", prog_name="cvdatakit")
def cli() -> None:
    """cvdatakit – Dataset-centric CV toolkit.\n
    Run dataset statistics, annotation checks, mislabel detection, and
    active-learning analysis on COCO-format datasets.
    """


# ── stats subcommand ──────────────────────────────────────────────────────────

@cli.command("stats")
@click.argument("annotation_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--image-dir", "-i", default=None, help="Root directory of images.")
@click.option(
    "--output", "-o", default=None,
    help="Save JSON summary to this path.",
)
@click.option("--tail-pct", default=10.0, show_default=True,
              help="Percentile below which a category is 'tail'.")
def stats_cmd(
    annotation_file: str,
    image_dir: Optional[str],
    output: Optional[str],
    tail_pct: float,
) -> None:
    """Print dataset statistics for a COCO annotation file."""
    from cvdatakit.io.coco_reader import COCODataset
    from cvdatakit.stats.dataset_stats import DatasetStats

    console.print(Panel(f"[bold cyan]cvdatakit stats[/] – {annotation_file}"))

    ds = COCODataset(annotation_file, image_dir=image_dir)
    stats = DatasetStats(ds)
    summary = stats.summary()
    summary["tail_categories"] = stats.tail_categories(tail_pct)

    # ── top-level numbers ─────────────────────────────────────────────────────
    t = Table(box=box.SIMPLE_HEAVY, show_header=False)
    t.add_column("Metric", style="bold")
    t.add_column("Value", style="cyan")
    t.add_row("Images", str(summary["num_images"]))
    t.add_row("Categories", str(summary["num_categories"]))
    t.add_row("Annotations", str(summary["num_annotations"]))
    ann = summary["annotations_per_image"]
    t.add_row(
        "Annotations/image",
        f"{ann['mean']:.2f} ± {ann['std']:.2f}  (min {ann['min']}, max {ann['max']})",
    )
    imb = summary["class_imbalance"]
    t.add_row("Gini imbalance", f"{imb.get('gini', 'n/a'):.4f}")
    t.add_row("Imbalance ratio", f"{imb.get('imbalance_ratio', 'n/a'):.1f}×")
    t.add_row("Tail categories", str(len(summary["tail_categories"])))
    console.print(t)

    # ── class distribution table ──────────────────────────────────────────────
    dist = stats.class_distribution()[:20]
    dt = Table("Category", "Count", "Fraction", box=box.MINIMAL_DOUBLE_HEAD, title="Class Distribution (top 20)")
    for row in dist:
        bar = "#" * int(row["fraction"] * 40)
        dt.add_row(row["name"], str(row["count"]), f"{row['fraction']:.3f} {bar}")
    console.print(dt)

    if summary["tail_categories"]:
        console.print(
            f"[yellow]Tail categories ({tail_pct:.0f}th pct):[/] "
            + ", ".join(summary["tail_categories"][:10])
            + ("…" if len(summary["tail_categories"]) > 10 else "")
        )

    if output:
        Path(output).write_text(json.dumps(summary, indent=2, default=str))
        console.print(f"[green]Saved JSON → {output}[/]")


# ── check subcommand ──────────────────────────────────────────────────────────

@cli.command("check")
@click.argument("annotation_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--min-bbox-area", default=4.0, show_default=True,
              help="Min bbox pixel area to flag as tiny.")
@click.option("--max-iou", default=0.85, show_default=True,
              help="IoU threshold for near-duplicate detection.")
@click.option("--output", "-o", default=None, help="Save issue list to JSON.")
def check_cmd(
    annotation_file: str,
    min_bbox_area: float,
    max_iou: float,
    output: Optional[str],
) -> None:
    """Run annotation-quality checks on a COCO annotation file."""
    from cvdatakit.io.coco_reader import COCODataset
    from cvdatakit.quality.annotation_checks import AnnotationChecker

    console.print(Panel(f"[bold cyan]cvdatakit check[/] – {annotation_file}"))

    ds = COCODataset(annotation_file)
    checker = AnnotationChecker(ds, min_bbox_area=min_bbox_area, max_overlap_iou=max_iou)
    summary = checker.summary()

    t = Table("Issue Type", "Count", box=box.SIMPLE_HEAVY, title="Annotation Issues")
    severity_color = {"error": "red", "warning": "yellow", "info": "blue"}
    issues = checker.run()
    by_type: dict = {}
    by_sev: dict = {}
    for iss in issues:
        by_type.setdefault(iss.issue_type, []).append(iss)
        by_sev[iss.severity] = by_sev.get(iss.severity, 0) + 1

    for issue_type, issue_list in sorted(by_type.items(), key=lambda x: -len(x[1])):
        sev = issue_list[0].severity
        color = severity_color.get(sev, "white")
        t.add_row(f"[{color}]{issue_type}[/]", str(len(issue_list)))

    console.print(t)

    sev_t = Table("Severity", "Count", box=box.SIMPLE_HEAVY)
    for sev, cnt in sorted(by_sev.items()):
        color = severity_color.get(sev, "white")
        sev_t.add_row(f"[{color}]{sev}[/]", str(cnt))
    console.print(sev_t)

    console.print(f"Total issues: [bold]{summary['total_issues']}[/]")

    if output:
        Path(output).write_text(json.dumps(summary, indent=2, default=str))
        console.print(f"[green]Saved JSON → {output}[/]")


# ── report subcommand (full pipeline) ─────────────────────────────────────────

@cli.command("report")
@click.argument("annotation_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--image-dir", "-i", default=None, help="Root directory of images.")
@click.option("--output-dir", "-o", default=None,
              help="Directory to write report files (default: annotation file directory).")
@click.option("--name", "-n", default="COCO", show_default=True, help="Dataset name.")
@click.option("--tail-pct", default=10.0, show_default=True,
              help="Percentile threshold for tail categories.")
def report_cmd(
    annotation_file: str,
    image_dir: Optional[str],
    output_dir: Optional[str],
    name: str,
    tail_pct: float,
) -> None:
    """Generate a full HTML + JSON report for a COCO-format dataset."""
    from cvdatakit.recipes.coco import COCORecipe

    console.print(Panel(f"[bold cyan]cvdatakit report[/] – {annotation_file}"))

    recipe = COCORecipe(
        annotation_file,
        image_dir=image_dir,
        report_dir=output_dir,
        dataset_name=name,
        tail_percentile=tail_pct,
    )
    result = recipe.run(save_report=True)

    console.print(f"[green]JSON report → {result.get('report_json', 'n/a')}[/]")
    console.print(f"[green]HTML report → {result.get('report_html', 'n/a')}[/]")

    # Print headline numbers
    stats = result.get("stats", {})
    console.print(
        f"\n[bold]Images:[/] {stats.get('num_images')}  "
        f"[bold]Categories:[/] {stats.get('num_categories')}  "
        f"[bold]Annotations:[/] {stats.get('num_annotations')}"
    )
    imb = stats.get("class_imbalance", {})
    if imb:
        console.print(
            f"[bold]Gini:[/] {imb.get('gini', '?'):.4f}  "
            f"[bold]Imbalance ratio:[/] {imb.get('imbalance_ratio', '?'):.1f}×  "
            f"[bold]Tail:[/] {len(stats.get('tail_categories', []))} categories"
        )

    checks = result.get("annotation_checks", {})
    if checks:
        console.print(
            f"[bold]Annotation issues:[/] {checks.get('total_issues', 0)}"
        )


# ── imagenet subcommand ───────────────────────────────────────────────────────

@cli.command("imagenet")
@click.argument("root_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", "-o", default=None,
              help="Directory to write report files.")
@click.option("--name", "-n", default="ImageNet", show_default=True)
@click.option("--max-per-class", default=None, type=int,
              help="Cap images scanned per class.")
def imagenet_cmd(
    root_dir: str,
    output_dir: Optional[str],
    name: str,
    max_per_class: Optional[int],
) -> None:
    """Run statistics on an ImageNet flat-folder dataset."""
    from cvdatakit.recipes.imagenet import ImageNetRecipe

    console.print(Panel(f"[bold cyan]cvdatakit imagenet[/] – {root_dir}"))

    recipe = ImageNetRecipe(
        root_dir,
        report_dir=output_dir,
        dataset_name=name,
        max_images_per_class=max_per_class,
    )
    result = recipe.run(save_report=True)
    stats = result.get("stats", {})
    imb = stats.get("class_imbalance", {})
    console.print(
        f"[bold]Images:[/] {stats.get('num_images')}  "
        f"[bold]Classes:[/] {stats.get('num_categories')}"
    )
    if imb:
        console.print(
            f"[bold]Gini:[/] {imb.get('gini', '?'):.4f}  "
            f"[bold]Imbalance ratio:[/] {imb.get('imbalance_ratio', '?'):.1f}×"
        )
    console.print(f"[green]HTML report → {result.get('report_html', 'n/a')}[/]")


if __name__ == "__main__":
    cli()
