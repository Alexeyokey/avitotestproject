"""Plot completed reranker audits without reading models or candidate caches.

The bundled Windows runtime has Pillow but no matplotlib. A small common canvas
renders identical geometry to a high-resolution PNG and an editable vector SVG.
Only complete-pool macro Recall@50 is plotted; selected stages come from tune.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT, SCALE = 1800, 1000, 2
INK = "#17252d"
MUTED = "#53646e"
GRID = "#dde5e9"
TRAIN = "#0072B2"
VALIDATION = "#D55E00"
SELECTION = "#7d8991"
MODEL_REPORTS = (
    ("hgb-report.json", "HistGradientBoosting", 150),
    ("lightgbm400-report.json", "LightGBM · классификация", 400),
    ("lambdarank400-report.json", "LightGBM · LambdaRank", 400),
)


def font_path(bold: bool = False) -> Path:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [Path("C:/Windows/Fonts") / name]
    if not bold:
        candidates.append(
            Path("C:/Users/alex_/.cache/codex-runtimes/codex-primary-runtime")
            / "dependencies/native/poppler/Library/share/fonts/DejaVuSans.ttf"
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cyrillic font unavailable: {name}")


class Canvas:
    """Write shapes to Pillow and SVG using the same logical coordinates."""

    def __init__(self) -> None:
        self.image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
            '<rect width="100%" height="100%" fill="white"/>',
        ]
        self.fonts: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}

    def font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(str(font_path(bold)), size * SCALE)
        return self.fonts[key]

    def text(self, x, y, text, size=20, color=INK, bold=False, anchor="start"):
        font = self.font(size, bold)
        width = self.draw.textlength(text, font=font) / SCALE
        adjusted_x = x - width / 2 if anchor == "middle" else x - width if anchor == "end" else x
        self.draw.text((adjusted_x * SCALE, y * SCALE), text, font=font, fill=color, anchor="lt")
        weight = "bold" if bold else "normal"
        self.svg.append(
            f'<text x="{x:.3f}" y="{y:.3f}" fill="{color}" '
            f'font-family="DejaVu Sans, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" dominant-baseline="text-before-edge">'
            f'{html.escape(text)}</text>'
        )

    def line(self, points, color=INK, width=2, dashed=False):
        scaled = [(x * SCALE, y * SCALE) for x, y in points]
        if dashed:
            # Dashed segments are used only for straight vertical selection lines.
            (x1, y1), (x2, y2) = points
            assert x1 == x2
            for y in range(round(y1), round(y2), 13):
                self.draw.line([(x1 * SCALE, y * SCALE), (x2 * SCALE, min(y + 7, y2) * SCALE)], fill=color, width=width * SCALE)
        else:
            self.draw.line(scaled, fill=color, width=width * SCALE, joint="curve")
        vertices = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
        dash = ' stroke-dasharray="7 6"' if dashed else ""
        self.svg.append(f'<polyline points="{vertices}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{dash}/>')

    def circle(self, x, y, radius, color, outline="white", width=1):
        self.draw.ellipse(((x - radius) * SCALE, (y - radius) * SCALE, (x + radius) * SCALE, (y + radius) * SCALE), fill=color, outline=outline, width=width * SCALE)
        self.svg.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius}" fill="{color}" stroke="{outline}" stroke-width="{width}"/>')

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        png = directory / "reranker_recall_curves.png"
        svg = directory / "reranker_recall_curves.svg"
        self.image.save(png, dpi=(300, 300))
        svg.write_text("\n".join(self.svg + ["</svg>"]), encoding="utf-8")
        return png, svg


def read_reports(directory: Path):
    result = []
    for filename, label, max_stage in MODEL_REPORTS:
        report = json.loads((directory / filename).read_text(encoding="utf-8"))
        stages = report["stages"]
        selected = report["selection"]["stage_selected_on_tuning_texts"]
        assert selected in stages, f"Unplotted selected stage in {filename}"
        assert stages == sorted(set(stages)), f"Unordered stages in {filename}"
        assert stages[-1] == max_stage, f"Incomplete audit in {filename}"
        assert report["historical_holdout_reused"] is True
        curves = {}
        counts = {}
        for dataset in ("train_sample", "validation"):
            rows = [report["datasets"][dataset][str(stage)]["all_groups"] for stage in stages]
            counts[dataset] = rows[0]["groups"]
            assert all(row["groups"] == counts[dataset] for row in rows)
            curves[dataset] = [row["slices"]["all"]["reranker_macro_recall"] for row in rows]
            assert all(0 <= value <= 1 for value in curves[dataset])
        assert counts == {"train_sample": 1000, "validation": 5157}, counts
        selected_index = stages.index(selected)
        result.append({
            "report": filename, "label": label, "model_class": report["model_class"],
            "stages": stages, "selected_stage_on_tune": selected,
            "train_sample_macro_recall": curves["train_sample"],
            "all_validation_macro_recall": curves["validation"],
            "selected_all_validation_macro_recall": curves["validation"][selected_index],
            "selected_tune_macro_recall": report["selection"]["tune"]["slices"]["all"]["reranker_macro_recall"],
            "selected_confirmation_macro_recall": report["selection"]["confirmation"]["slices"]["all"]["reranker_macro_recall"],
            "groups": counts,
        })
    return result


def plot(models, directory: Path):
    canvas = Canvas()
    canvas.text(65, 34, "Реранкер: Recall@50 по числу деревьев", 34, bold=True)
    canvas.text(65, 86, "Полные фиксированные пулы · fit sample: 1 000 групп · строгая validation: 5 157 групп", 22, color=MUTED)
    for x, color, text in [(65, TRAIN, "Обучающая сторона: выборка 1 000"), (690, VALIDATION, "Вся строгая validation")]:
        canvas.line([(x, 144), (x + 45, 144)], color, 3)
        canvas.circle(x + 22, 144, 5, color)
        canvas.text(x + 59, 130, text, 21)
    canvas.line([(1260, 134), (1260, 154)], SELECTION, 2, dashed=True)
    canvas.text(1274, 130, "Этап выбран только по tune", 21)

    top, bottom, y_min, y_max = 300, 718, 0.837, 0.883
    ticks = [0.84, 0.85, 0.86, 0.87, 0.88]
    for index, model in enumerate(models):
        left, right = 95 + 570 * index, 600 + 570 * index
        max_x = 160 if index == 0 else 410
        px = lambda stage: left + stage / max_x * (right - left)
        py = lambda recall: bottom - (recall - y_min) / (y_max - y_min) * (bottom - top)
        stage = model["selected_stage_on_tune"]
        selected_value = model["selected_all_validation_macro_recall"]
        canvas.text(left, 205, model["label"], 23, bold=True)
        canvas.text(left, 243, f"Выбор tune: {stage} деревьев · validation {selected_value * 100:.2f}%".replace(".", ","), 19, color=MUTED)
        for value in ticks:
            y = py(value)
            canvas.line([(left, y), (right, y)], GRID, 1)
            if index == 0:
                canvas.text(left - 15, y - 12, f"{value * 100:.0f}%", 19, anchor="end")
        canvas.line([(left, top), (left, bottom), (right, bottom)], MUTED, 1)
        xticks = [0, 50, 100, 150] if index == 0 else [0, 100, 200, 300, 400]
        for tick in xticks:
            x = px(tick)
            canvas.line([(x, bottom), (x, bottom + 6)], MUTED, 1)
            canvas.text(x, bottom + 16, str(tick), 19, anchor="middle")
        canvas.line([(px(stage), top), (px(stage), bottom)], SELECTION, 2, dashed=True)
        for field, color in [("train_sample_macro_recall", TRAIN), ("all_validation_macro_recall", VALIDATION)]:
            vertices = [(px(s), py(value)) for s, value in zip(model["stages"], model[field])]
            canvas.line(vertices, color, 3)
            for x, y in vertices:
                canvas.circle(x, y, 5, color)
        canvas.circle(px(stage), py(selected_value), 9, VALIDATION, INK, 2)
        canvas.text((left + right) / 2, 774, "Число деревьев", 21, anchor="middle")

    canvas.text(65, 837, "Recall включает положительные объявления вне пула; fit sample содержит такие группы.", 19, color=MUTED)
    canvas.text(65, 872, "Историческая validation уже использовалась в экспериментах; это не новая независимая проверка.", 19, color=MUTED)
    canvas.text(65, 907, "Энкодер обучался на fit-текстах: разрыв train / validation сам по себе не доказывает переобучение.", 19, color=MUTED)
    canvas.text(65, 960, "Источник: завершённые JSON-аудиты · macro Recall@50 · 12 октября 2026", 16, color=MUTED)
    png, svg = canvas.save(directory)
    (directory / "reranker_recall_curves.data.json").write_text(json.dumps({
        "metric": "all_groups.slices.all.reranker_macro_recall",
        "stage_selection": "tune only", "historical_holdout_reused": True,
        "models": models,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return png, svg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/research-2026-10-12/curves"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research-2026-10-12/figures"))
    args = parser.parse_args()
    models = read_reports(args.input_dir)
    png, svg = plot(models, args.output_dir)
    print(json.dumps({"png": str(png.resolve()), "svg": str(svg.resolve()), "models": models}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
