#!/usr/bin/env python3
"""Parse or replay the SIFT1M proxy trace and generate its paper figure."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from generate_paper_ivf_assets import PdfCanvas, write_pdf


TIME_RE = re.compile(r"^\s*([^:]+):\s+([0-9.]+)\s+s$")
RECON_RE = re.compile(r"^\s*reconstruction error \(sample\):\s+([0-9.]+)$")
QPS_RE = re.compile(r"^\s*QPS:\s+([0-9.]+)$")
RECALL_RE = re.compile(
    r"^\s*recall@1:\s+([0-9.]+)\s+recall@10:\s+([0-9.]+)\s+"
    r"recall@100:\s+([0-9.]+)\s+recall@1000:\s+([0-9.]+)$"
)
OVERLAP_RE = re.compile(r"^\s*overlap@1000\(gt=1000\):\s+([0-9.]+)$")
EXIT_RE = re.compile(r"^\s*Exit status:\s+([0-9]+)$")
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_manifest(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records[int(record["id"])] = record
    return records


def parse_log(path: Path) -> dict:
    out: dict[str, object] = {"log": str(path)}
    with path.open() as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("meta.run "):
                meta = json.loads(line[len("meta.run ") :])
                out["maxtrain"] = meta.get("maxtrain")
                out["target"] = ",".join(meta.get("targets", []))
                out["epq_structure"] = meta.get("epq_structure")
            elif line.startswith("meta.dataset "):
                meta = json.loads(line[len("meta.dataset ") :])
                out["train_rows"] = meta.get("train_rows")
                out["base_rows"] = meta.get("base_rows")
                out["query_rows"] = meta.get("query_rows")
            elif line.startswith("meta.build "):
                meta = json.loads(line[len("meta.build ") :])
                out["structure_trace_enabled"] = meta.get("structure_trace_enabled")
            elif line.startswith("meta.method "):
                meta = json.loads(line[len("meta.method ") :])
                main = meta.get("main", {})
                builder = main.get("builder", {})
                out["main_bits"] = meta.get("main_bits")
                out["tail_bits"] = meta.get("tail_bits")
                out["tail_stages"] = meta.get("tail_stages")
                out["builder_name"] = builder.get("name")
                out["builder_type"] = builder.get("type")
                out["builder_group_count"] = builder.get("group_count")
                out["tail_alt_best_mse"] = meta.get("tail_alt_best_mse")
                out["tail_alt_final_mse"] = meta.get("tail_alt_final_mse")
            else:
                m = TIME_RE.match(line)
                if m:
                    key = m.group(1).strip().lower().replace("/", "_").replace(" ", "_")
                    if key == "training_total":
                        key = "train_total"
                    out[key] = float(m.group(2))
                    continue
                m = RECON_RE.match(line)
                if m:
                    out["reconstruction_error"] = float(m.group(1))
                    continue
                m = QPS_RE.match(line)
                if m:
                    out["qps"] = float(m.group(1))
                    continue
                m = RECALL_RE.match(line)
                if m:
                    out["recall1"] = float(m.group(1))
                    out["recall10"] = float(m.group(2))
                    out["recall100"] = float(m.group(3))
                    out["recall1000"] = float(m.group(4))
                    continue
                m = OVERLAP_RE.match(line)
                if m:
                    out["overlap1000"] = float(m.group(1))
                    continue
                m = EXIT_RE.match(line)
                if m:
                    out["exit_status"] = int(m.group(1))
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=float), kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rankdata(xs).tolist(), rankdata(ys).tolist())


def hex_rgb(text: str) -> tuple[float, float, float]:
    value = text.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def nice_ticks(lo: float, hi: float, target: int = 5) -> tuple[float, float, list[float]]:
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError("axis bounds must be finite")
    if lo == hi:
        pad = 0.5 if lo == 0 else abs(lo) * 0.05
        lo -= pad
        hi += pad
    span = hi - lo
    raw_step = span / max(target - 1, 1)
    base = 10 ** math.floor(math.log10(raw_step))
    step = base
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        candidate = multiple * base
        if candidate >= raw_step:
            step = candidate
            break
    axis_lo = math.floor(lo / step) * step
    axis_hi = math.ceil(hi / step) * step
    ticks: list[float] = []
    value = axis_lo
    limit = axis_hi + 0.5 * step
    while value <= limit:
        ticks.append(0.0 if abs(value) < 1e-12 else value)
        value += step
    return axis_lo, axis_hi, ticks


def format_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def draw_marker(
    canvas: PdfCanvas,
    x: float,
    y: float,
    shape: str,
    color: tuple[float, float, float],
    size: float = 2.7,
) -> None:
    canvas.marker(x, y, shape, size, color)


def draw_panel(
    canvas: PdfCanvas,
    *,
    left: float,
    bottom: float,
    width: float,
    height: float,
    x_values: list[float],
    y_values: list[float],
    rows: list[dict],
    y_key: str,
    y_label: str,
    stats_label: str,
    stage_styles: dict[str, tuple[tuple[float, float, float], str, str]],
    x_axis: tuple[float, float, list[float]],
    y_axis: tuple[float, float, list[float]],
) -> None:
    x_lo, x_hi, x_ticks = x_axis
    y_lo, y_hi, y_ticks = y_axis

    def map_x(value: float) -> float:
        return left + (value - x_lo) / (x_hi - x_lo) * width

    def map_y(value: float) -> float:
        return bottom + (value - y_lo) / (y_hi - y_lo) * height

    canvas.stroke_rgb((0.84, 0.84, 0.84))
    canvas.line_width(0.35)
    for tick in x_ticks:
        x = map_x(tick)
        canvas.line(x, bottom, x, bottom + height)
    for tick in y_ticks:
        y = map_y(tick)
        canvas.line(left, y, left + width, y)

    canvas.stroke_rgb((0.0, 0.0, 0.0))
    canvas.line_width(0.7)
    canvas.line(left, bottom, left + width, bottom)
    canvas.line(left, bottom, left, bottom + height)

    for tick in x_ticks:
        x = map_x(tick)
        canvas.stroke_rgb((0.0, 0.0, 0.0))
        canvas.line_width(0.55)
        canvas.line(x, bottom, x, bottom - 2.0)
        canvas.fill_rgb((0.0, 0.0, 0.0))
        canvas.text_center(x, bottom - 11.0, format_tick(tick), 6.5)
    for tick in y_ticks:
        y = map_y(tick)
        canvas.stroke_rgb((0.0, 0.0, 0.0))
        canvas.line_width(0.55)
        canvas.line(left - 2.0, y, left, y)
        canvas.fill_rgb((0.0, 0.0, 0.0))
        canvas.text_right(left - 5.0, y - 2.2, format_tick(tick), 6.5)

    coef = np.polyfit(np.asarray(x_values), np.asarray(y_values), 1)
    line_y0 = float(np.polyval(coef, x_lo))
    line_y1 = float(np.polyval(coef, x_hi))
    canvas.stroke_rgb((0.10, 0.10, 0.10))
    canvas.line_width(1.05)
    canvas.line(map_x(x_lo), map_y(line_y0), map_x(x_hi), map_y(line_y1))

    for row in rows:
        source = str(row["source"])
        if source not in stage_styles:
            continue
        color, shape, _ = stage_styles[source]
        draw_marker(
            canvas,
            map_x(float(row["j_star"]) / 10000.0),
            map_y(float(row[y_key])),
            shape,
            color,
        )

    canvas.fill_rgb((0.0, 0.0, 0.0))
    canvas.text(left, bottom + height + 7.0, y_label, 7.7, "F2")
    canvas.text(left + 6.0, bottom + height - 12.0, stats_label, 6.8)
    canvas.text_center(left + width / 2.0, 10.5, "Partition-search proxy J*(P), x1e4", 7.3)


def write_png_from_pdf(pdf_path: Path, png_path: Path) -> None:
    if shutil.which("pdftoppm") is None:
        return
    png_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "figure"
        subprocess.run(
            ["pdftoppm", "-singlefile", "-png", "-r", "240", str(pdf_path), str(prefix)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        shutil.copyfile(prefix.with_suffix(".png"), png_path)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "id",
        "stage",
        "source",
        "step",
        "j_star",
        "groups",
        "builder_group_count",
        "train_rows",
        "maxtrain",
        "main_bits",
        "tail_bits",
        "tail_stages",
        "train_total",
        "add_encode_time",
        "search_time",
        "qps",
        "recall1",
        "recall10",
        "recall100",
        "recall1000",
        "overlap1000",
        "reconstruction_error",
        "tail_alt_best_mse",
        "exit_status",
        "structure",
        "log",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            record = {key: row.get(key, "") for key in fields}
            for key in ("structure", "log"):
                value = str(record[key])
                if value:
                    candidate = Path(value)
                    if candidate.is_absolute():
                        try:
                            record[key] = str(candidate.relative_to(REPO_ROOT))
                        except ValueError:
                            pass
            writer.writerow(record)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_figure(pdf_path: Path, png_path: Path | None, rows: list[dict], summary: dict) -> None:
    stage_styles = {
        "grow": (hex_rgb("#4C78A8"), "circle", "ClusterGrow"),
        "crystallize": (hex_rgb("#F58518"), "square", "Crystallization"),
        "chain_tail": (hex_rgb("#54A24B"), "diamond", "Chained Marginal Refinement"),
    }
    x = [float(row["j_star"]) / 10000.0 for row in rows]
    y_recall = [float(row["recall10"]) for row in rows]
    y_recon = [float(row["reconstruction_error"]) for row in rows]

    width = 482.0
    height = 176.0
    canvas = PdfCanvas(width, height)

    legend_x = 44.0
    legend_y = height - 17.0
    for source in ("grow", "crystallize", "chain_tail"):
        color, shape, label = stage_styles[source]
        draw_marker(canvas, legend_x, legend_y + 2.6, shape, color, 2.7)
        canvas.fill_rgb((0.0, 0.0, 0.0))
        canvas.text(legend_x + 6.0, legend_y, label, 7.2)
        legend_x += 0.55 * 7.2 * len(label) + 23.0

    panel_left = 45.0
    panel_bottom = 32.0
    panel_gap = 47.0
    panel_width = (width - panel_left - 13.0 - panel_gap) / 2.0
    panel_height = 98.0
    x_axis = nice_ticks(min(x), max(x), target=5)
    recall_axis = nice_ticks(min(y_recall), max(y_recall), target=5)
    recon_axis = nice_ticks(min(y_recon), max(y_recon), target=5)

    draw_panel(
        canvas,
        left=panel_left,
        bottom=panel_bottom,
        width=panel_width,
        height=panel_height,
        x_values=x,
        y_values=y_recall,
        rows=rows,
        y_key="recall10",
        y_label="Recall@10",
        stats_label=(
            f"Pearson {summary['pearson_jstar_recall10']:.3f}, "
            f"Spearman {summary['spearman_jstar_recall10']:.3f}"
        ),
        stage_styles=stage_styles,
        x_axis=x_axis,
        y_axis=recall_axis,
    )
    draw_panel(
        canvas,
        left=panel_left + panel_width + panel_gap,
        bottom=panel_bottom,
        width=panel_width,
        height=panel_height,
        x_values=x,
        y_values=y_recon,
        rows=rows,
        y_key="reconstruction_error",
        y_label="Final reconstruction error",
        stats_label=(
            f"Pearson {summary['pearson_jstar_recon']:.3f}, "
            f"Spearman {summary['spearman_jstar_recon']:.3f}"
        ),
        stage_styles=stage_styles,
        x_axis=x_axis,
        y_axis=recon_axis,
    )

    write_pdf(pdf_path, width, height, canvas.stream())
    if png_path is not None:
        write_png_from_pdf(pdf_path, png_path)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/benchmark_tables/tmp_jstar_sift128_arepq_trace/manifest.jsonl"),
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("docs/benchmark_tables/tmp_jstar_sift128_arepq_eval"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("docs/benchmark_tables/jstar_sift128_arepq_proxy.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("docs/benchmark_tables/jstar_sift128_arepq_proxy_summary.json"),
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("../paper/generated/jstar-sift128-arepq-proxy.pdf"),
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=Path("docs/benchmark_tables/jstar_sift128_arepq_proxy.png"),
    )
    parser.add_argument(
        "--reuse-csv",
        action="store_true",
        help="regenerate summary/figures from --csv without raw manifests or logs",
    )
    args = parser.parse_args()

    if args.reuse_csv:
        rows: list[dict] = load_csv(args.csv)
    else:
        manifest = load_manifest(args.manifest)
        rows = []
        for log_path in sorted(args.eval_dir.glob("id_*.log")):
            match = re.search(r"id_(\d+)\.log$", log_path.name)
            if not match:
                continue
            trace_id = int(match.group(1))
            if trace_id not in manifest:
                raise RuntimeError(f"missing manifest record for id={trace_id}")
            row = dict(manifest[trace_id])
            parsed = parse_log(log_path)
            row.update(parsed)
            row["structure"] = row.get("path")
            if row.get("exit_status") != 0:
                raise RuntimeError(f"{log_path} did not finish successfully")
            if row.get("builder_type") != "fixed":
                raise RuntimeError(f"{log_path} did not use FixedStructureBuilder")
            if row.get("maxtrain") != 0 or row.get("train_rows") != 100000:
                raise RuntimeError(f"{log_path} did not use full SIFT1M train protocol")
            rows.append(row)
    rows.sort(key=lambda row: int(row["id"]))
    if len(rows) != 16:
        raise RuntimeError(f"expected 16 proxy-trace rows, found {len(rows)}")

    x = [float(row["j_star"]) for row in rows]
    recall10 = [float(row["recall10"]) for row in rows]
    recall1 = [float(row["recall1"]) for row in rows]
    recon = [float(row["reconstruction_error"]) for row in rows]
    overlap = [float(row["overlap1000"]) for row in rows]
    summary = {
        "n": len(rows),
        "ids": [int(row["id"]) for row in rows],
        "j_host_min": min(x),
        "j_host_max": max(x),
        "recall10_min": min(recall10),
        "recall10_max": max(recall10),
        "reconstruction_error_min": min(recon),
        "reconstruction_error_max": max(recon),
        "pearson_jstar_recall10": pearson(x, recall10),
        "spearman_jstar_recall10": spearman(x, recall10),
        "pearson_jstar_recall1": pearson(x, recall1),
        "spearman_jstar_recall1": spearman(x, recall1),
        "pearson_jstar_overlap1000": pearson(x, overlap),
        "spearman_jstar_overlap1000": spearman(x, overlap),
        "pearson_jstar_recon": pearson(x, recon),
        "spearman_jstar_recon": spearman(x, recon),
    }

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    if not args.reuse_csv:
        write_csv(args.csv, rows)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    plot_figure(args.pdf, args.png, rows, summary)

    print(json.dumps(summary, indent=2))
    print(f"{'reused' if args.reuse_csv else 'wrote'} {args.csv}")
    print(f"wrote {args.summary}")
    print(f"wrote {args.pdf}")
    print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
