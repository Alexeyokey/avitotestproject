# Обучение, артефакты и повторный запуск

Инструкции соответствуют коду на 14 октября 2026. [README](../README.md)
содержит короткий запуск, [EXPERIMENTS.md](EXPERIMENTS.md) — результаты и решения.

## 1. Обычный CLI

После установки проекта доступны:

| Команда | Назначение | Основной результат |
| --- | --- | --- |
| `profile` | Профиль исходных данных | JSON через `--output` |
| `evaluate` | Локальный Recall и диагностика | JSON через `--output` |
| `predict` | Все benchmark-запросы | CSV через `--answer` |
| `validate` | Проверка готового CSV | Результат проверки |
| `fit-microcategory` | Мягкий классификатор с OOF | Модель через `--microcategory-model` |
| `prepare-reranker` | Обучающие признаки на кандидатах | Parquet через `--reranker-data` |
| `fit-reranker` | HGB / LightGBM / LambdaRank | Модель через `--reranker-model` |

`fit-reranker` печатает JSON в stdout; его `--output` сейчас не сохраняет отчёт.
Для лога используйте перенаправление. В командах оценки/подготовки явно задавайте
`--split queries --seed 42`; обычное значение split по умолчанию — `pairs`.

Пример обучения микрокатегорий:

```bash
avito fit-microcategory --data-dir dataset \
  --split queries --seed 42 --microcategory-crossfit-folds 3 \
  --microcategory-model artifacts/microcategory-oof3.joblib \
  --output artifacts/microcategory-fit.json
```

Для V8 подготовка требует hybrid, сохранённого OOF-классификатора,
положительного `--microcategory-candidate-k`, `--rescore-dense-union`
и `--reranker-feature-version 8`. `--supplemental-fill micro-typed-dense-v1`
требует micro K998, поле `combined`, top-3, формат документов `current`
и положительные веса BM25/dense. Полные флаги смотрите в `avito --help`.

Пример fit уже подготовленных признаков:

```bash
avito fit-reranker --reranker-data artifacts/reranker-train.parquet \
  --reranker-model artifacts/reranker.joblib \
  --reranker-estimator histgb --reranker-max-iter 150 \
  > artifacts/reranker-fit.log 2>&1
```

Путь модели реранкера сам по себе не определяет поиск. Для inference/evaluate
должны совпадать корпус, схема признаков, encoder, split и retrieval-конфигурация.
Нельзя удалять поля metadata или отключать проверку совместимости для переноса
модели на другой пул.

## 2. Границы последнего исследовательского маршрута

Последние geo-dense и регуляризационные опыты находятся в ветке
`research/microcategory-2026-10-13`. Ветка содержит код опытов и документацию;
большие модели, исходные данные и индексы хранятся отдельно от Git.

Обычный CLI не содержит флага геоприора перед dense top-K.
`dense_retrieval_policy` и `geo_dense_*` новой модели не совпадают с обычным
`retrieval_config`. Обычный `predict` и `export_offline_solution.py` такие настройки
отклоняют. Поддерживаемые старые ordinary/V7/V8 inference-пути остаются отдельными.

Текущие модели связаны с каталогом 200 000 и исходным train. Их нельзя напрямую
применить к оригинальным 189 212 объявлениям benchmark. Для нового `answer.csv`
нужны интеграция этого поиска и отдельная подготовка под целевой корпус.

Ниже — повторение **существующего локального исследования**, а не полный запуск
из чистого Git-клона. Большие данные, модели и индексы не публикуются в Git.
Все исходные файлы перечислены в manifests и отчётах. При переносе файлов
нужно согласовать пути и сохранить проверяемое содержимое.

## 3. Окружение текущего WSL

```bash
cd /home/alex/avitotestproject-reranker-new-octen
export PYTHONPATH=src:scripts
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
PYTHON=/home/alex/avitotestproject/.venv/bin/python
```

Этот Python установлен в основной WSL-копии. `PYTHONPATH` направляет импорты
на исследовательский checkout. Если вы создали новое окружение в текущем
проекте, задайте его интерпретатор вместо указанного пути.

Для исследования нужны уже подготовленные:

| Каталог артефактов | Содержимое |
| --- | --- |
| `fresh-validation-2026-10-13` | Fit/Tune/Audit, каталог, manifests и V4-контроль |
| `microcategory-research-2026-10-13` | OOF-классификатор, V8-признаки/priors, HGB-100, исходный dense-кеш |
| `geo-dense-research-2026-10-14` | Прежние corpus/query-векторы и результаты географии |
| `geo-dense-reranker-2026-10-14` | Новые fit-признаки, географический HGB-150, выбранные 50, кеш Tune |
| `geo-dense-regularization-2026-10-14` | План двух опытов, новые HGB-150, результаты и выбранные 50 |

Они находятся в `artifacts/` относительно checkout. В текущем WSL этот каталог
совместно используется с основной копией проекта.

## 4. Повторение географии и переобучения

`prepare_geo_dense_vectors.py` извлекает существующие corpus-векторы и кодирует
запросы; аргументы и идентичности входов сохранены в `vectors/manifest.json`.
Это не новый embedding-fit. `prepare_geo_dense_fit_vectors.py` аналогично
подготавливает query-вектора обучающих групп.

`prepare_geo_dense_reranker.py` создаёт новую таблицу fit на новых географических
пулах, сохраняя прежний выбор 30 000 групп и sampler. Результат:

```text
artifacts/geo-dense-reranker-2026-10-14/
  train-geo005-v8.parquet
  train-geo005-v8.parquet.json
  train-geo005-v8.parquet.priors.joblib
  hgb-geo005-v8-150.joblib
  full-tune/report.json
  full-tune/group-scores.jsonl
  full-tune/tune-features.parquet
  full-tune/tune-features.parquet.json
  full-tune/hgb-geo005-selected50.joblib
```

Оценка полной обученной модели на новом каталоге результатов:

```bash
bash scripts/run_geo_dense_retrained_eval.sh \
  artifacts/repeated-geo-tune \
  --bootstrap-replicates 2000 --with-training-loss
```

Обёртка содержит абсолютные пути данного WSL и требует прежние контрольные
отчёты/модели. Существующие каталоги не перезаписываются.
`run_geo_dense_retraining_after_prepare.sh` ждёт PID уже запущенной подготовки;
это оркестратор конкретного продолжения, а не самостоятельная команда с нуля.

Независимая проверка завершённой основной оценки:

```bash
"$PYTHON" scripts/summarize_geo_dense_retraining.py \
  --report artifacts/geo-dense-reranker-2026-10-14/full-tune/report.json \
  --output-dir artifacts/repeated-geo-summary
```

Сумматор проверяет полные контексты, знаменатели, выбор префикса,
перезагрузку модели и кеш. Выходной каталог также должен быть новым.

## 5. Два опыта регуляризации на сохранённых признаках

Для простого повторения **оценки уже обученных** вариантов:

```bash
REG=artifacts/geo-dense-regularization-2026-10-14

"$PYTHON" scripts/evaluate_geo_dense_regularization.py \
  --experiment-plan "$REG/experiment-plan.json" \
  --protocol-dir artifacts/fresh-validation-2026-10-13/protocol \
  --maxfeatures-model "$REG/hgb-geo005-v8-maxfeatures08-150.joblib" \
  --leaves-model "$REG/hgb-geo005-v8-leaves15-150.joblib" \
  --output-dir "$REG/evaluation-repeat" \
  --bootstrap-replicates 10000 --save-selected-models
```

Новый каталог `evaluation-repeat` обязателен. Повторяется fixed50-сравнение
с исходным географическим HGB-50; текущие файлы не подменяются.

Для повторения **обучения и оценки** двух отдельных вариантов:

```bash
REG=artifacts/geo-dense-regularization-repeat
mkdir "$REG"
cp artifacts/geo-dense-regularization-2026-10-14/experiment-plan.json \
  "$REG/experiment-plan.json"

"$PYTHON" scripts/train_geo_dense_regularization.py fit \
  --variant both --output-dir "$REG" > "$REG/fit.log" 2>&1

"$PYTHON" scripts/evaluate_geo_dense_regularization.py \
  --experiment-plan "$REG/experiment-plan.json" \
  --protocol-dir artifacts/fresh-validation-2026-10-13/protocol \
  --maxfeatures-model "$REG/hgb-geo005-v8-maxfeatures08-150.joblib" \
  --leaves-model "$REG/hgb-geo005-v8-leaves15-150.joblib" \
  --output-dir "$REG/full-tune" \
  --bootstrap-replicates 10000 --save-selected-models \
  > "$REG/evaluate.log" 2>&1

"$PYTHON" scripts/summarize_geo_dense_regularization.py \
  --experiment-dir "$REG" --output-dir artifacts/regularization-repeat-summary
```

По умолчанию trainer читает исходную географическую HGB-150 и её точную
обучающую таблицу. Копирование плана сохраняет зафиксированные входы и критерии;
если они изменились, проверки SHA останавливают запуск.

Trainer клонирует все параметры, меняя только `max_features=0.8` **или**
`max_leaf_nodes=15`. Варианты обучаются последовательно, по 150 деревьев,
на 1 935 640 строках 85 признаков и прежних весах. Случайное разбиение строк
для early stopping не используется. Первичное сравнение —50 против 50;
75/100/150 и выбор минимального префикса в пределах 0,2 п.п. — вторичный анализ.

Кеш Tune содержит **4 740 500 строк /4 750 доступных групп**, имеет роль
`tune-only-never-training` и не используется для fit. Оценщик сохраняет
все 12 103 контекста, проверяет pool IDs/порядок/метки/знаменатели и воспроизводит
прежние четыре префикса на каждой группе. Tie: вероятность → RRF → исходная
позиция кандидата. Поиск, индексы и embeddings не пересчитываются.

## 6. Сохранённый runtime-префикс

Физически `.joblib` содержит все 150 обученных деревьев. Выбранный артефакт
дополнительно хранит `selected_iteration=50`. `Reranker.rank_batch` и
`prediction_scores(..., selected_iteration=50)` учитывают его.
Прямой `model.predict_proba()` использует все 150 и даст результат другого префикса.

Лучший отдельный файл:
`artifacts/geo-dense-regularization-2026-10-14/full-tune/hgb-geo005-leaves15-selected50.joblib`.
Его `.joblib.json` связывает параметры, fitted fingerprint, исходный train,
отчёт и retrieval-конфигурацию. Текущая ранее выбранная модель не заменена.

## 7. Автономный пакет обычного инференса

`scripts/export_offline_solution.py` поддерживает обычные сохранённые конфигурации,
включая проверенные V7/V8 microcategory и supplemental-настройки. Созданный пакет
содержит проект, веса/локальный HF-кеш, версии зависимостей и `run_predict.py`:

```bash
python run_predict.py --data-dir /absolute/path/to/dataset \
  --device cuda --embedding-batch-size 32 --output answer.csv
```

`run_predict.py` создаётся экспортёром и не является файлом корня этого репозитория.
Данные предоставляются отдельно; на новом компьютере HNSW-индекс строится локально
при первом запуске. Совместимость корпуса и train всё равно проверяется.
Этот пример относится к поддерживаемому обычному пакету; последний geo-dense
кандидат пока требует отдельной интеграции.

## 8. Проверки

```bash
python -m unittest discover -s tests -q
PYTHONPATH=src:scripts "$PYTHON" -m unittest \
  scripts.test_train_geo_dense_regularization \
  scripts.test_evaluate_geo_dense_regularization -q
```

Для текущих двух опытов прошли 13 целевых тестов. Проверяются единственное
изменение параметра, запрещённое использование Tune для fit, разрыв query-групп
между блоками Parquet, tie, контрольный префикс, строки путей и перезагрузка
выбранного артефакта. Более ранние проверки поиска и протокола описаны
в соответствующих отчётах в `docs/`.
