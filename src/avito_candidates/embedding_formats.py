"""Experimental, explicit input layouts for embedding retrieval.

These functions do not change the production DenseRetriever or reuse a corpus
cache under another format. A document format change requires a new index.
Filter parsing is deliberately conservative: only named semantic fields are
selected, and unrecognized free text is retained rather than silently lost.
"""
from __future__ import annotations

import re

from .core import normalize
from .dense import item_text, query_text

FORMAT_VERSION = 1

# These are metadata field names, not learned category labels or query answers.
# Longest-first matching prevents "тип услуги" consuming "тип услуги автосервиса".
SEMANTIC_KEYS = (
    "тип услуги автосервиса", "предмет или специальность", "марка автомобиля",
    "тип автомобиля", "чем вы занимаетесь", "тип оборудования", "тип техники",
    "название услуги", "специализация", "вид услуги", "тип услуги",
    "вид работы", "вид работ", "тип работ", "назначение", "услуга",
)
OTHER_KEYS = (
    "рейтинг пользователя", "место оказания услуг", "минимальная сумма заказа",
    "тип стоимости за услугу", "тип стоимости за урок", "тип стоимости за час",
    "тип стоимости за м²", "тип стоимости за м2", "тип стоимости",
    "график работы от", "график работы до", "время работы, с", "время работы, до",
    "работаете с юрлицами и ип", "наличие медицинского образования",
    "опыт работы", "как вы работаете", "куда выезжаете", "работа по договору",
    "гарантия на работу", "начальная цена", "стоимость", "продолжительность",
    "рабочие дни", "учебное учреждение", "год окончания", "преподаватель",
    "занятия", "доставка", "залог", "дни", "район", "адрес",
)
_ALL_KEYS = sorted(set(SEMANTIC_KEYS + OTHER_KEYS), key=lambda k: (-len(k), k))
_KEY_PATTERN = re.compile(r"(?<!\S)(?:" + "|".join(map(re.escape, _ALL_KEYS)) + r")(?=\s|$)")


def extract_filter_values(text):
    """Select service/type values from Avito's linear key/value metadata.

No parsed semantic key: retain unknown free text. If all recognized fields
are nonsemantic (e.g. a rating floor), return empty; they belong in explicit
ranking features. Unknown leading free text is preserved. An unknown key
inside a value remains in that value, avoiding a destructive guess.
"""
    text = normalize(text or "")
    if not text:
        return ""
    matches = list(_KEY_PATTERN.finditer(text))
    if not matches:
        return text
    values = []
    leading = text[:matches[0].start()].strip()
    if leading:
        values.append(leading)
    for i, match in enumerate(matches):
        stop = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = text[match.end():stop].strip()
        if match.group(0) in SEMANTIC_KEYS and value:
            values.append(value)
    return " ; ".join(dict.fromkeys(values))


def query_format(query, variant="current"):
    if variant == "current":
        return query_text(query)
    text = normalize(query.get("search_query", ""))
    if variant == "plain":
        return text
    if variant == "values":
        values = extract_filter_values(query.get("search_infm_params_text", ""))
        return text + " | " + values if text and values else text or values
    if variant == "no_category":
        return query_text({**query, "search_category": ""})
    raise ValueError(f"Unknown query format: {variant}")


def item_format(item, variant="compact", *, params_words=40, description_words=48):
    if params_words < 0 or description_words < 0:
        raise ValueError("Field word limits must be nonnegative")
    if variant == "current":
        return item_text(item, params_words=params_words, description_words=description_words)
    title = normalize(item.get("item_title_raw", ""))
    if variant == "title_description":
        description = normalize(str(item.get("item_description_raw", "") or "")[:400])
        return ". ".join(value for value in (title, description) if value)
    if variant == "compact":
        values = " ".join(extract_filter_values(item.get("item_infm_params_text", "")).split()[:params_words])
        description = " ".join(normalize(item.get("item_description_raw", "")).split()[:description_words])
        return ". ".join(value for value in (title, values, description) if value)
    raise ValueError(f"Unknown item format: {variant}")
