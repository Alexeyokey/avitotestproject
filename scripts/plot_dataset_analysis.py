"""Рисует в Matplotlib сравнение train, локальной оценки и тестовых данных."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


COLORS = ("#4567A7", "#087F78", "#B85E18")


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#17253A",
        "text.color": "#17253A",
        "axes.edgecolor": "#CBD5E1",
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def _finish(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E3E9F0", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.tick_params(axis="x", colors="#54657A")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def _save_svg(fig, output: Path) -> None:
    """Убирает пробелы в путях Matplotlib, чтобы Git не считал их ошибкой."""
    fig.savefig(output, format="svg", metadata={"Date": None})
    lines = output.read_text(encoding="utf-8").splitlines()
    output.write_text("\n".join(line.rstrip() for line in lines) + "\n",
                      encoding="utf-8")
    plt.close(fig)


def query_shift(summary: dict, output: Path) -> None:
    """Сравнивает доли признаков, явно указывая единицу каждой выборки."""
    pops = summary["populations"]
    assert pops["train_events"]["count"] == 497673
    assert pops["local_validation_groups"]["count"] == 5157
    assert pops["benchmark_queries"]["count"] == 2452
    series = [
        ("Train: события", pops["train_events"]),
        ("Локальная оценка: группы", pops["local_validation_groups"]),
        ("Тест: запросы", pops["benchmark_queries"]),
    ]
    labels = ("Есть фильтры", "Нет точной локации в корпусе")
    features = ("with_filter_fraction", "no_exact_location_fraction")
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    for index, (name, data) in enumerate(series):
        values = [data[key] for key in features]
        bars = ax.barh(y + (index - 1) * 0.23, values, height=0.19,
                       label=name, color=COLORS[index])
        for bar, value in zip(bars, values):
            ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2,
                    f"{value * 100:.1f}%".replace(".", ","), va="center",
                    fontsize=10, fontweight="semibold")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.82)
    ax.set_xticks(np.arange(0, 0.81, 0.2))
    ax.set_title("Признаки запросов: локальная оценка и тест", loc="left", pad=29)
    ax.legend(loc="upper center", bbox_to_anchor=(0.55, 1.10),
              ncol=3, frameon=False, fontsize=9)
    _finish(ax)
    fig.text(0.02, 0.025,
             "Train взвешен по событиям; локальная оценка — по группам полного запроса. "
             "У теста нет открытых меток.", fontsize=9, color="#54657A")
    fig.subplots_adjust(left=0.31, right=0.96, top=0.77, bottom=0.14)
    _save_svg(fig, output)


def overlap(summary: dict, output: Path) -> None:
    """Показывает покрытие между train и тестом без предположений о релевантности."""
    data = summary["overlap"]
    rows = [
        ("Текст тестового запроса есть в train",
         data["benchmark_texts_seen_in_train"], COLORS[2]),
        ("Полный тестовый запрос есть в train",
         data["benchmark_full_queries_seen_in_train"], COLORS[2]),
        ("Объявление корпуса есть в train",
         data["corpus_items_seen_in_train"], COLORS[0]),
        ("Строка train с объявлением из корпуса",
         data["train_rows_with_item_in_corpus"], COLORS[0]),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    y = np.arange(len(rows))
    values = [row[1]["fraction"] for row in rows]
    bars = ax.barh(y, values, height=0.53, color=[row[2] for row in rows])
    for bar, (_, metric, _) in zip(bars, rows):
        value = metric["fraction"]
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{value * 100:.1f}%".replace(".", ","), va="center",
                fontsize=11, fontweight="semibold")
    ax.set_yticks(y, [row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.49)
    ax.set_xticks(np.arange(0, 0.51, 0.1))
    ax.set_title("Пересечение train с тестовыми данными", loc="left", pad=20)
    _finish(ax)
    fig.text(0.02, 0.025,
             "Доли считаются от разных наборов: 2 452 запроса, 189 212 объявлений "
             "корпуса и 497 673 строк train.", fontsize=9, color="#54657A")
    fig.subplots_adjust(left=0.43, right=0.96, top=0.85, bottom=0.14)
    _save_svg(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path,
                        default=Path("docs/dataset-analysis.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("docs/figures"))
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    query_shift(summary, args.output_dir / "dataset-query-shift.svg")
    overlap(summary, args.output_dir / "dataset-overlap.svg")
    print(args.output_dir)


if __name__ == "__main__":
    main()
