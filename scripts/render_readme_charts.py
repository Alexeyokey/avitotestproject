"""Строит SVG-графики для README из проверенных результатов экспериментов."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


BLUE = "#4567A7"
TEAL = "#087F78"
ORANGE = "#B85E18"
TEXT = "#17253A"
MUTED = "#54657A"
GRID = "#DDE4EC"


def label(x: float, y: float, value: str, *, size: int = 15,
          fill: str = TEXT, anchor: str = "start", weight: int = 400) -> str:
    """Подписывает точку SVG с экранированием пользовательского текста."""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}">{escape(value)}</text>')


def rectangle(x: float, y: float, width: float, height: float, fill: str,
              radius: int = 4) -> str:
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
            f'height="{height:.1f}" rx="{radius}" fill="{fill}"/>')


def line(x1: float, y1: float, x2: float, y2: float, color: str = GRID) -> str:
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
            f'y2="{y2:.1f}" stroke="{color}" stroke-width="1"/>')


def svg(width: int, height: int, title: str, description: str,
        content: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="title desc">\n'
            f'<title id="title">{escape(title)}</title>\n'
            f'<desc id="desc">{escape(description)}</desc>\n'
            f'{rectangle(0, 0, width, height, "#FFFFFF", 0)}\n'
            + "\n".join(content) + "\n</svg>\n")


def percent(value: float) -> str:
    return f"{value * 100:.2f}".replace(".", ",") + "%"


def geo_chart(data: dict) -> str:
    """Сравнивает два прогона BM25 при неизменных настройках, кроме геоканала."""
    assert data["split"] == "queries" and data["seed"] == 42
    assert data["candidate_k"] == 500 and data["evaluated_queries"] == 5157
    assert data["location_bonus"] == 0.1
    assert data["geographic"]["geo_candidate_k"] == 100
    control, geographic = data["control"], data["geographic"]

    title = "Вклад географических каналов BM25"
    description = ("Локальная проверка на 5157 группах. Recall пула вырос с "
                   "84,22% до 94,72%, Recall@50 — с 70,59% до 76,99%.")
    parts = [label(36, 47, title, size=26, weight=700),
             label(36, 76, "split=queries · seed=42 · k=500 · бонус локации 0,1",
                   size=14, fill=MUTED)]
    left, span = 225, 665
    for tick in range(0, 101, 20):
        x = left + span * tick / 100
        parts += [line(x, 122, x, 354),
                  label(x, 384, f"{tick}%", size=12, fill=MUTED, anchor="middle")]

    parts += [rectangle(225, 95, 17, 17, BLUE, 2),
              label(250, 109, "Глобальный поиск", size=14),
              rectangle(429, 95, 17, 17, TEAL, 2),
              label(454, 109, "С географическими каналами", size=14)]
    rows = [
        ("Recall пула", "candidate_pool_recall", 153),
        ("Recall@50", "recall_at_50", 271),
    ]
    for name, key, y in rows:
        parts.append(label(36, y + 36, name, size=19, weight=600))
        for data, color, offset in ((control, BLUE, 0), (geographic, TEAL, 36)):
            value = data[key]
            x_end = left + span * value
            parts += [rectangle(left, y + offset, span * value, 25, color),
                      label(x_end + 7, y + offset + 19, percent(value),
                            size=14, weight=600)]
    return svg(960, 405, title, description, parts)


def octen_chart(data: dict) -> str:
    """Отображает результаты трёх одинаковых проверок Octen на другом ПК."""
    # Настройки и метрики записаны рядом с README в docs/readme-results.json.
    assert data["split"] == "queries" and data["seed"] == 42
    assert data["candidate_k"] == 499 and data["evaluated_queries"] == 5157
    assert data["location_bonus"] == 0.02
    experiments = [
        (row["name"], (row["dense_recall"], row["candidate_pool_recall"],
                       row["recall_at_50"]))
        for row in data["experiments"]
    ]
    assert len(experiments) == 3
    title = "Эффект дообучения Octen"
    description = ("На одинаковой локальной проверке Recall dense-канала, "
                   "всего пула и итогового топ-50 растёт от исходной модели "
                   "к пилоту и полной LoRA-модели.")
    parts = [label(36, 47, title, size=26, weight=700),
             label(36, 76, "BM25 по описанию + dense · k=499 · бонус 0,02 · 5 157 групп",
                   size=14, fill=MUTED)]
    legend = [(BLUE, "Dense"), (TEAL, "Пул"), (ORANGE, "Топ-50")]
    for index, (color, name) in enumerate(legend):
        x = 340 + index * 130
        parts += [rectangle(x, 92, 16, 16, color, 2),
                  label(x + 24, 106, name, size=14)]

    left, top, bottom, plot_height = 78, 135, 421, 286
    for tick in range(0, 101, 20):
        y = bottom - plot_height * tick / 100
        parts += [line(left, y, 920, y),
                  label(left - 12, y + 5, f"{tick}%", size=12,
                        fill=MUTED, anchor="end")]
    colors = (BLUE, TEAL, ORANGE)
    for group_index, (name, values) in enumerate(experiments):
        center = 222 + group_index * 280
        for metric_index, (value, color) in enumerate(zip(values, colors)):
            x = center - 78 + metric_index * 55
            height = plot_height * value
            y = bottom - height
            parts += [rectangle(x, y, 42, height, color),
                      label(x + 21, y - 8, percent(value), size=13,
                            weight=600, anchor="middle")]
        parts.append(label(center, 459, name, size=17, weight=600,
                           anchor="middle"))
    return svg(960, 485, title, description, parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.repo_root
    output = args.output_dir or root / "docs" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    results = json.loads((root / "docs" / "readme-results.json").read_text(encoding="utf-8"))
    (output / "geo-recall.svg").write_text(geo_chart(results["bm25_geo"]), encoding="utf-8")
    (output / "octen-recall.svg").write_text(octen_chart(results["octen"]), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
