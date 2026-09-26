# Поиск объявлений для запросов Авито

Проект решает задачу кандидатогенерации: для каждого поискового запроса нужно
выбрать до 50 объявлений из корпуса. Качество оценивается по Recall@50.

В проекте есть три режима поиска:

- `bm25` отдельно ищет по заголовкам, параметрам и описаниям;
- `dense` ищет близкие по смыслу объявления с помощью эмбеддингов;
- `hybrid` объединяет результаты BM25 и эмбеддингов через RRF.

В ответ попадают до 50 лучших объявлений.

## Установка

Нужен Python 3.11 или новее.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

В Windows окружение активируется командой `.venv\Scripts\Activate.ps1`.

Для режимов `dense` и `hybrid` установите дополнительные зависимости:

```bash
pip install -e '.[dense]'
```

Положите `train.parquet`, `benchmark_queries.parquet` и
`benchmark_items.parquet` в папку `dataset`. Вместо этого можно передать
абсолютный путь к папке с данными через `--data-dir`.

## Команды

Запустить тесты:

```shell
python -m unittest discover -s tests -v
```

Посмотреть основные пересечения между обучающей выборкой и бенчмарком:

```shell
avito profile --data-dir dataset --output artifacts/profile.json
```

Проверить качество на отложенных парах «запрос — объявление»:

```shell
avito evaluate --data-dir dataset --split pairs --output artifacts/bm25-pairs.json
```

Проверить качество на текстах запросов, которых модель не видела при обучении:

```shell
avito evaluate --data-dir dataset --split queries --output artifacts/bm25-queries.json
```

Для быстрого пробного запуска можно добавить `--max-queries 200`.

Три поля индексируются отдельно и объединяются через RRF. Настройки по умолчанию:

| Поле | `k1` | `b` | Вес |
|---|---:|---:|---:|
| Заголовок | 1.0 | 0.2 | 2.0 |
| Параметры | 1.2 | 0.7 | 1.0 |
| Описание | 1.0 | 0.8 | 0.5 |

Настройки меняются через `--title-k1`, `--title-b`, `--title-weight` и такие же
аргументы с префиксами `params` и `description`. `--k1` и `--b` переопределяют
значение сразу для всех трёх полей. Внутри BM25 используется `--bm25-rrf-k 30`
и квота `--bm25-channel-quota 10`.

Первый запуск семантического поиска скачивает модель
`intfloat/multilingual-e5-small`, кодирует корпус и сохраняет HNSW-индекс в
`artifacts/dense`. Следующие команды используют готовый индекс. На Mac с Apple
Silicon можно передать `--device mps`; если этот режим работает нестабильно,
используйте `--device cpu`.

Сначала отдельно измерьте dense-поиск:

```bash
avito evaluate --data-dir dataset --split queries --method dense --device mps \
  --output artifacts/dense-queries.json
```

Затем проверьте объединение с BM25:

```bash
avito evaluate --data-dir dataset --split queries --method hybrid --device mps \
  --output artifacts/hybrid-queries.json
```

Гибридный поиск берёт по 300 кандидатов каждого канала. RRF объединяет их с
равными весами, при этом первые 10 результатов каждого канала сохраняют место
в итоговых 50. Параметры можно менять через `--candidate-k`, `--rrf-k`,
`--bm25-weight`, `--dense-weight` и `--channel-quota`.

Собрать и проверить файл для отправки:

```shell
avito predict --data-dir dataset --method hybrid --device mps \
  --answer artifacts/answer.csv
avito validate --data-dir dataset --answer artifacts/answer.csv
```

## Как устроена проверка качества

В режиме `pairs` одинаковые пары «запрос — объявление» всегда попадают в одну
часть выборки. Это защищает оценку от прямого повторения одной и той же пары.

В режиме `queries` целиком откладываются тексты запросов вместе со всеми
локациями и фильтрами. Такая проверка показывает, как поиск работает на новых
формулировках.

Индекс BM25 строится по корпусу объявлений. Данные из train используются для проверки:
Recall@50 сначала считается отдельно для каждого запроса, а затем усредняется.

Проект использует pandas, PyArrow, NumPy и scikit-learn. Все вычисления
выполняются локально.
