"""Independently reconstruct fixed50 metrics and package the two tests."""
from pathlib import Path
import argparse
import gzip
import json
import math
import shutil
import subprocess

from summarize_geo_dense_retraining import aggregate, sha256, verify_summary

STAGES = (50, 75, 100, 150)
VARIANTS = ('maxfeatures08', 'leaves15')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.experiment_dir, args.output_dir
    if output.exists():
        raise FileExistsError(output)
    report_path = root / 'full-tune/report.json'
    report = read(report_path)
    assert report['complete'] and report['full_tune_complete'] and report['groups'] == 12103
    assert report['primary_stage'] == report['primary_baseline_stage'] == 50
    assert not report['audit_labels_read'] and not report['retrieval_rebuilt'] and not report['inventory_reencoded']
    score_path = root / 'full-tune/group-scores.jsonl'
    assert sha256(score_path) == report['group_scores_sha256']
    records = [json.loads(line) for line in score_path.read_text(encoding='utf-8').splitlines()]
    plan = read(root / 'experiment-plan.json')
    old_scores = Path(plan['previous_group_scores']['path'])
    assert sha256(old_scores) == plan['previous_group_scores']['sha256']
    old = [json.loads(line)['new'] for line in old_scores.read_text(encoding='utf-8').splitlines()]
    assert len(records) == len(old) == 12103
    baseline = []
    variants = {key: [] for key in VARIANTS}
    hits = {f's{stage}_hits' for stage in STAGES}
    for record, reference in zip(records, old):
        assert record['group_identity'] == reference['group_identity']
        assert record['baseline'] == reference
        baseline.append(reference)
        assert set(record['variants']) == set(VARIANTS)
        for name in VARIANTS:
            row = record['variants'][name]
            assert {key: value for key, value in row.items() if key not in hits} == {
                key: value for key, value in reference.items() if key not in hits}
            assert row['pool_size'] <= 998
            for stage in STAGES:
                value = row[f's{stage}_hits']
                assert isinstance(value, int) and 0 <= value <= min(50, row['pool_hits'])
            variants[name].append(row)
    verify_summary(baseline, report['baseline'], 50)
    base_metric = aggregate(baseline, 's50_hits', 'available_relevant_count')
    assert base_metric['groups'] == 4750 and base_metric['texts'] == 3069
    assert math.isclose(base_metric['text_macro_recall_at_50'], .8902455738261614, abs_tol=1e-12)
    results = {}
    for name, rows in variants.items():
        values = {}
        for stage in STAGES:
            verify_summary(rows, report['secondary_capacity_curves'][name][str(stage)], stage)
            value = aggregate(rows, f's{stage}_hits', 'available_relevant_count')['text_macro_recall_at_50']
            values[stage] = value
            pair = report['secondary_paired_vs_current50'][name][str(stage)]
            assert math.isclose(pair['delta'], value - base_metric['text_macro_recall_at_50'], abs_tol=1e-12)
            assert pair['replicates'] == 10000 and pair['query_text_clusters'] == 3069
        assert report['primary_fixed50'][name]['summary'] == report['secondary_capacity_curves'][name]['50']
        assert report['primary_fixed50'][name]['paired_vs_current50'] == report['secondary_paired_vs_current50'][name]['50']
        chosen = min(stage for stage, value in values.items() if max(values.values()) - value <= .002 + 1e-12)
        assert chosen == report['secondary_selected_stages'][name]
        fit_path = root / f'hgb-geo005-v8-{name}-150.json'
        fit = read(fit_path)
        assert fit['complete'] and sha256(Path(fit['model'])) == fit['model_sha256']
        changed = {key: value for key, value in fit['experiment']['new_parameters'].items()
                   if fit['experiment']['source_parameters'][key] != value}
        assert changed == plan['variants'][name]
        results[name] = {'primary_recall_at_50': values[50],
            'primary_paired_comparison': report['primary_fixed50'][name]['paired_vs_current50'],
            'capacity_curve': values, 'secondary_selected_stage': chosen,
            'secondary_selected_recall': values[chosen],
            'full_truth_recall_at_50': aggregate(rows, 's50_hits', 'full_relevant_count')['text_macro_recall_at_50'],
            'fit_seconds': fit['experiment']['fit_seconds']}
    parity = report['cache_parity']
    assert parity['rows'] == 4740500 and parity['available_groups'] == 4750 and parity['all_groups'] == 12103
    assert parity['zero_availability_groups'] == 7353 and parity['pool_id_order_and_labels_verified']
    assert parity['source_stages_reproduced_per_context'] == list(STAGES)
    summary = {'verified': True, 'primary_stage': 50, 'baseline_recall_at_50': base_metric['text_macro_recall_at_50'],
        'variants': results, 'groups': 12103, 'available_groups': 4750, 'available_texts': 3069,
        'pool_recall': aggregate(baseline, 'pool_hits', 'available_relevant_count')['text_macro_recall_at_50'],
        'maximum_pool_size': 998, 'input_report_sha256': sha256(report_path),
        'current_model_replaced': False, 'audit_opened': False,
        'note': 'Fixed50 primary comparisons; secondary capacity choice remains separate. Reused Tune is exploratory.'}
    lines = ['Регуляризация географического реранкера — 14 октября 2026', '',
        'Основное сравнение: одинаковые 50 деревьев, один и тот же Tune и кеш признаков.',
        'Те же обучающие строки, веса, признаки, геопул и seed42. Каждый опыт меняет один параметр.',
        f'Текущий HGB50: Recall@50 = {100*summary["baseline_recall_at_50"]:.5f}%.',
        f'Recall пула = {100*summary["pool_recall"]:.5f}%; во всех контекстах 998 кандидатов.', '',
        'Вариант                      Recall@50%  Разница, п.п.   95% исследовательский интервал']
    for name, result in results.items():
        paired = result['primary_paired_comparison']
        low, high = paired['percentile_95_interval']
        lines.append(f'{name:28} {100*result["primary_recall_at_50"]:10.5f} {100*paired["delta"]:+12.5f}   [{100*low:+.5f}; {100*high:+.5f}]')
    lines += ['', 'Дополнительная кривая сложности, Recall@50%:', 'Вариант                       50         75        100        150    Выбранный префикс']
    for name, result in results.items():
        lines.append(f'{name:28} ' + ' '.join(f'{100*result["capacity_curve"][stage]:10.5f}' for stage in STAGES)
                     + f'    {result["secondary_selected_stage"]}')
    lines += ['', 'Вторичный выбор: минимальное число деревьев в пределах 0,2 п.п. от лучшего варианта своей кривой.',
        'Его результат не подменяет основное сравнение при фиксированных 50 деревьях.', '',
        'Проверки:',
        '- Метрики независимо пересчитаны по всем 12103 контекстам; доступны положительные у 4750 контекстов / 3069 текстов.',
        '- Recall усредняется по нормализованным текстам запросов и считается по доступным положительным объявлениям.',
        '- Все 4 префикса прежней модели точно воспроизвели число попаданий на каждой группе кеша.',
        '- Проверены идентификаторы, порядок кандидатов, метки и полные знаменатели.',
        '- Использовано 4740500 строк сохранённых признаков Tune только для оценки.',
        '- Поиск, эмбеддинги и приоры не пересчитывались; Audit не открывался.', '',
        'Ограничения:',
        '- Tune уже многократно использовался. Интервалы номинальные 95%, без поправки на множественные сравнения.',
        '- Это исследовательский корпус 200000 объявлений, не новая внешняя оценка соревнования.',
        '- Текущая модель не заменена. Новые артефакты сохраняются отдельно.',
        '- Для официального benchmark по-прежнему нужна интеграция нового геопоиска и проверка переноса.']
    text = '\n'.join(lines) + '\n'
    output.mkdir(parents=True)
    (output / 'FINAL_REPORT.txt').write_text(text, encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for relative in ('experiment-plan.json', 'fit.log', 'evaluate.log', 'full-tune/report.json', 'full-tune/declaration.json',
                     *(f'hgb-geo005-v8-{name}-150.json' for name in VARIANTS)):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    for manifest in (root / 'full-tune').glob('*.joblib.json'):
        shutil.copy2(manifest, output / 'full-tune' / manifest.name)
    with score_path.open('rb') as source, gzip.open(output / 'full-tune/group-scores.jsonl.gz', 'wb') as destination:
        shutil.copyfileobj(source, destination)
    subprocess.run(['git', 'diff', '--cached', '--check'], check=True)
    tree = subprocess.check_output(['git', 'write-tree'], text=True).strip()
    archive = output / f'source-tree-{tree[:12]}.zip'
    subprocess.run(['git', 'archive', '--format=zip', '--output=' + str(archive.resolve()), tree], check=True)
    state = {'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'tree': tree, 'committed': False, 'source_archive': archive.name}
    (output / 'CODE_STATE.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    manifest = {str(path.relative_to(output)): {'bytes': path.stat().st_size, 'sha256': sha256(path)}
                for path in sorted(output.rglob('*')) if path.is_file()}
    (output / 'PACKAGE_SHA256.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
