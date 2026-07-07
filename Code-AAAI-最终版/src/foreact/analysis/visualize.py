"""Dependency-free SVG visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


COLORS = ["#2f6f9f", "#d17b0f", "#3a8f5b", "#8a5fbf", "#b84545"]


def write_depth_svg(path: str | Path, curves: Mapping[str, Mapping[str, float]]) -> None:
    width, height = 720, 420
    margin = 54
    depths = sorted({int(depth) for curve in curves.values() for depth in curve})
    if not depths:
        depths = [0, 1]
    min_d, max_d = min(depths), max(depths)

    def x(depth: int) -> float:
        if max_d == min_d:
            return margin
        return margin + (depth - min_d) / (max_d - min_d) * (width - 2 * margin)

    def y(value: float) -> float:
        return height - margin - max(0.0, min(1.0, value)) * (height - 2 * margin)

    parts = [_svg_header(width, height), _axes(width, height, margin, "PlanDepth SR(d)")]
    for idx, (method, curve) in enumerate(curves.items()):
        color = COLORS[idx % len(COLORS)]
        points = [(x(int(depth)), y(float(value))) for depth, value in sorted(curve.items(), key=lambda item: int(item[0]))]
        if points:
            path_data = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{path_data}" />')
            for px, py in points:
                parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}" />')
            parts.append(f'<text x="{width - margin - 160}" y="{margin + 22 * idx}" font-size="13" fill="{color}">{method}</text>')
    parts.append("</svg>\n")
    _write(path, "\n".join(parts))


def write_efficiency_svg(path: str | Path, metrics_by_method: Mapping[str, Mapping[str, object]]) -> None:
    width, height = 720, 420
    margin = 58
    methods = list(metrics_by_method)
    max_success = max([float(metrics_by_method[m].get("success_rate", 0.0)) for m in methods] + [1.0])
    bar_width = (width - 2 * margin) / max(1, len(methods) * 1.6)
    parts = [_svg_header(width, height), _axes(width, height, margin, "Success vs. Overhead")]
    for idx, method in enumerate(methods):
        metrics = metrics_by_method[method]
        success = float(metrics.get("success_rate", 0.0))
        extra = float(metrics.get("extra_forward_per_task", 0.0))
        x0 = margin + idx * bar_width * 1.6 + bar_width * 0.3
        h = success / max_success * (height - 2 * margin)
        y0 = height - margin - h
        color = COLORS[idx % len(COLORS)]
        parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="{color}" opacity="0.85" />')
        parts.append(f'<text x="{x0:.1f}" y="{height - 30}" font-size="11" transform="rotate(-25 {x0:.1f},{height - 30})">{method}</text>')
        parts.append(f'<text x="{x0:.1f}" y="{y0 - 8:.1f}" font-size="11">SR {success:.2f}, +F {extra:.1f}</text>')
    parts.append("</svg>\n")
    _write(path, "\n".join(parts))


def write_case_heatmap_svg(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    cell = 42
    label_w = 220
    top = 44
    depths = sorted({int(item["depth"]) for row in rows for item in row.get("forecast", [])})
    width = label_w + max(1, len(depths)) * cell + 40
    height = top + max(1, len(rows)) * cell + 50
    parts = [_svg_header(width, height)]
    parts.append('<text x="20" y="26" font-size="18" font-weight="700">PlanDepth Case Study Forecast Heatmap</text>')
    for col, depth in enumerate(depths):
        parts.append(f'<text x="{label_w + col * cell + 12}" y="{top - 10}" font-size="12">h={depth}</text>')
    for r, row in enumerate(rows):
        y0 = top + r * cell
        action = str(row.get("action", ""))
        parts.append(f'<text x="14" y="{y0 + 25}" font-size="12">{_escape(action[:34])}</text>')
        forecast = {int(item["depth"]): float(item["dead_end_mass"]) for item in row.get("forecast", [])}
        for col, depth in enumerate(depths):
            value = max(0.0, min(1.0, forecast.get(depth, 0.0)))
            red = int(235 * value + 245 * (1 - value))
            green = int(245 * (1 - value) + 95 * value)
            blue = int(245 * (1 - value) + 80 * value)
            x0 = label_w + col * cell
            parts.append(f'<rect x="{x0}" y="{y0}" width="{cell - 3}" height="{cell - 3}" fill="rgb({red},{green},{blue})" stroke="#ffffff" />')
            parts.append(f'<text x="{x0 + 8}" y="{y0 + 25}" font-size="11">{value:.2f}</text>')
    parts.append("</svg>\n")
    _write(path, "\n".join(parts))


def write_eld_svg(path: str | Path, rows: Sequence[Mapping[str, object]], title: str = "Effective Lookahead Depth") -> None:
    width, height = 760, 420
    margin = 58
    if not rows:
        _write(path, _svg_header(width, height) + "</svg>\n")
        return
    depths = sorted({int(row["depth"]) for row in rows})
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    max_signal = max([float(row.get("axis_signal_max", row.get("future_signal", 0.0))) for row in rows] + [1.0])
    min_d, max_d = min(depths), max(depths)

    def x(depth: int) -> float:
        if max_d == min_d:
            return margin
        return margin + (depth - min_d) / (max_d - min_d) * (width - 2 * margin)

    def y(value: float) -> float:
        return height - margin - max(0.0, min(max_signal, value)) / max_signal * (height - 2 * margin)

    parts = [_svg_header(width, height), _axes(width, height, margin, title)]
    for idx, method in enumerate(methods):
        color = COLORS[idx % len(COLORS)]
        method_rows = [row for row in rows if str(row["method"]) == method]
        points = [(x(int(row["depth"])), y(float(row["future_signal"]))) for row in method_rows]
        if not points:
            continue
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="'
            + " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
            + '" />'
        )
        eld = int(method_rows[0].get("effective_lookahead_depth", 0))
        if eld:
            eld_x = x(eld)
            parts.append(f'<line x1="{eld_x:.1f}" y1="{margin}" x2="{eld_x:.1f}" y2="{height - margin}" stroke="{color}" stroke-dasharray="5 5" />')
            parts.append(f'<text x="{eld_x + 4:.1f}" y="{margin + 18}" font-size="12" fill="{color}">ELD={eld}</text>')
        parts.append(f'<text x="{width - margin - 190}" y="{margin + 22 * idx}" font-size="13" fill="{color}">{_escape(method)}</text>')
    parts.append('<text x="60" y="390" font-size="12">Smoke proxy; full paper uses intervention-probe future signal.</text>')
    parts.append("</svg>\n")
    _write(path, "\n".join(parts))


def write_pareto_svg(path: str | Path, metrics_by_method: Mapping[str, Mapping[str, object]]) -> None:
    width, height = 760, 420
    margin = 64
    methods = list(metrics_by_method)
    max_extra = max([float(metrics_by_method[m].get("extra_forward_per_task", 0.0)) for m in methods] + [1.0])

    def x(extra: float) -> float:
        return margin + max(0.0, extra) / max_extra * (width - 2 * margin)

    def y(success: float) -> float:
        return height - margin - max(0.0, min(1.0, success)) * (height - 2 * margin)

    parts = [_svg_header(width, height), _axes(width, height, margin, "Success-Overhead Pareto")]
    for idx, method in enumerate(methods):
        metrics = metrics_by_method[method]
        success = float(metrics.get("success_rate", 0.0))
        extra = float(metrics.get("extra_forward_per_task", 0.0))
        color = COLORS[idx % len(COLORS)]
        px, py = x(extra), y(success)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="{color}" />')
        parts.append(f'<text x="{px + 9:.1f}" y="{py + 4:.1f}" font-size="12" fill="{color}">{_escape(method)}</text>')
    parts.append('<text x="60" y="390" font-size="12">x = extra forward count per task; Mode A should remain at zero overhead.</text>')
    parts.append("</svg>\n")
    _write(path, "\n".join(parts))


def _axes(width: int, height: int, margin: int, title: str) -> str:
    return "\n".join(
        [
            f'<text x="{margin}" y="28" font-size="18" font-weight="700">{title}</text>',
            f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" />',
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" />',
            f'<text x="{margin}" y="{height - 12}" font-size="12">depth / method</text>',
            f'<text x="10" y="{margin - 18}" font-size="12">success</text>',
        ]
    )


def _svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
