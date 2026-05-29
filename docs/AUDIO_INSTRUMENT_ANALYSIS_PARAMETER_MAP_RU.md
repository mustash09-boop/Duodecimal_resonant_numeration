# Карта параметров анализа аудио для определения инструмента

## Текущее состояние проекта

Этот документ фиксирует, **по каким параметрам мы уже можем анализировать аудио**, если цель состоит в том, чтобы различать инструменты не по одному спектральному отпечатку, а по многослойному поведению звука.

Он опирается на уже существующие блоки проекта:

- одиночные реальные инструменты:
  [BLOCK004_REAL_INSTRUMENTS_RESEARCH_RU.md](/E:/Duodecimal_resonant_numeration/docs/BLOCK004_REAL_INSTRUMENTS_RESEARCH_RU.md)
- ансамблевое поведение инструмента:
  [ENSEMBLE_INSTRUMENT_BEHAVIOR_HYPOTHESIS.md](/E:/Duodecimal_resonant_numeration/py/music12/blocks/Block002_pipeline/ENSEMBLE_INSTRUMENT_BEHAVIOR_HYPOTHESIS.md)

Главный смысл этой карты:

- показать, **что именно уже измеряется**;
- где это считается;
- что это означает физически или музыкально;
- и в каких режимах это особенно полезно:
  - одиночная нота,
  - полифония,
  - ансамбль,
  - будущая сепарация.

---

## 1. Крупные классы признаков

На текущем этапе у нас уже есть:

- **10 крупных классов признаков** для одиночных тональных инструментов;
- **15 крупных классов**, если включать ансамблевые роли и событийную онтологию;
- **30+ конкретных измеряемых параметров** внутри этих классов.

Это уже намного больше, чем обычная схема:

- “основная частота”
- “спектр”
- “тембр в среднем”

Наше текущее измерение инструмента включает:

- корень;
- гармоническую цепь;
- амплитуды гармоник;
- временную жизнь;
- инструментальную коробку;
- связь коробки с гармониками;
- пространственную геометрию;
- lineage-порождение резонансов;
- роль инструмента в событии;
- совместное владение событием.

---

## 2. Таблица параметров

Ниже:  
`Параметр / группа` → `Где считается` → `Что означает` → `Особенно полезно для`

| Параметр / группа | Где считается | Что означает | Особенно полезно для |
|---|---|---|---|
| `consensus_root_token` | root consensus summaries в `10_reports` | Какой корневой токен алгоритм считает причиной ноты | Одиночная нота, сравнение с эталоном |
| `consensus_root_hz` | root consensus summaries | Реальная частота корня, а не просто ожидаемая по имени файла | Настройка, дрейф, instrument realism |
| `root_delta_cents_vs_theory` | root consensus summaries, passport notes | Насколько реальная нота отклоняется от теоретической | Стабильность инструмента, интонация |
| `tuner_confidence` | root consensus summaries, passports | Насколько уверенно root подтверждается внутренней гармонической структурой | Надёжность root-идентичности |
| `present_harmonics` | root consensus summaries | Какие гармоники реально участвуют в жизни ноты | Гармоническая подпись инструмента |
| `member_count` / `unique_frame_count` | root consensus summaries | Насколько цепь велика и насколько долго держится | Устойчивость harmonic chain |
| Гармоническая цепь как класс | `dense_harmonic_chain_builder_cli`, root consensus | Нота понимается как связанная лестница гармоник, а не как один пик | Вся тональная магистраль |
| `theory_match_ratio` | dense-vs-theory, passports | Насколько очищенный dense согласуется с теоретическим гармоническим ожиданием | Проверка качества нотной модели |
| `mean_abs_delta_cents` | dense-vs-theory | Среднее расхождение с теорией по гармоническому составу | Отклонение реального инструмента от идеальной схемы |
| Глобальная частотная присутствуемость | `range_research_from_reports_cli.py` | Какие токены и частоты повторяются по многим нотам инструмента | Поиск instrument-wide behavior |
| `note_count` / `percent_notes` для cluster | `20_range_research/*dense_frequency_clusters.csv` | В скольких нотах живёт этот компонент | Общая коробка инструмента |
| `mean_amp`, `median_amp`, `max_amp` cluster | `20_range_research` | Средняя и пиковая сила повторяющихся компонентов | Сила instrument body |
| `box_all` | instrument passport JSON/MD | Полный слой повторяющихся dense-компонентов по инструменту | Общий отпечаток инструмента |
| `box_breath` | `split_box_layers_cli.py`, passport | Низкий дыхательный / механический слой | Корпус, механика, низ инструмента |
| `box_resonance` | `split_box_layers_cli.py`, passport | Основной резонансный слой тела инструмента | Идентичность корпуса и резонанса |
| `box_harmonic_relation` class | `box_harmonic_relation_cli.py` | Относится ли компонент к гармоническому узлу, близок к нему или нет | Связь между нотой и коробкой |
| `HARMONIC / NEAR_HARMONIC / NON_HARMONIC` | `box_harmonic_relation_cli.py` | Формальный тип связи box-компонента с гармоникой | Физическая интерпретация отклика |
| `harmonic_index` для relation | `box_harmonic_relation_cli.py` | К какой гармонике ближе всего тяготеет компонент | Какая часть ноты порождает box |
| `delta_cents` relation | `box_harmonic_relation_cli.py` | Точность привязки resonance к гармоническому узлу | Тонкость связности |
| `note_box_profile` как класс | `note_box_profile_builder_cli.py` | Собственная note-specific коробка отдельной ноты после вычитания общего box | Отличие конкретной ноты от общего тела инструмента |
| `frame_count`, `presence_ratio` note-box token | `30_note_box_profiles` | Насколько устойчиво note-box компонент держится у данной ноты | Локальная устойчивость ноты |
| `spiral12` coordinates | `spiral12_from_dense_clean_cli.py` | Геометрическое положение токена в 12-ричной спирали | Пространственная логика звука |
| `spiral3d` coordinates | `note_box_spiral3d_builder_cli.py` | Как нота, box и dense-other разворачиваются во времени | Видимость жизни ноты |
| `component_type` = `chain / note_box / dense_other` | `50_spiral3d/*points.csv` | К какому морфологическому слою принадлежит точка | Разделение ядра, коробки и остатка |
| `harmonic_index` in spiral3d | `note_box_spiral3d_builder_cli.py` | Какая гармоника соответствует точке core chain | Структура ядра ноты |
| `amplitude` / `relative_amp` | dense, spiral12, spiral3d, comparison tools | Насколько силён компонент относительно кадра или ноты | Тембровая иерархия |
| Амплитудный профиль гармоник | `multi_instrument_harmonic_amplitude_compare.py` | Как распределены силы одинаковых гармоник у разных инструментов | Отличие рояля, струнных, органа |
| `phase-aware harmonic amplitude signature` | частично уже исследовано в compare tools | Не только какие гармоники есть, но как их амплитуда живёт по фазам | Следующий сильный шаг для инструмента |
| `parent_harmonic_index` | `harmonic_chain_spiral3d_builder_cli.py` | Какая гармоника-ядро породила данную резонансную точку | Причинная lineage-структура |
| `parent_xy_distance` | `harmonic_chain_spiral3d_builder_cli.py` | Насколько близка точка к своему породившему harmonic core | Надёжность lineage-привязки |
| `lineage_role` = `harmonic_core / spawned_note_box / spawned_residual / unassigned_resonance` | `55_harmonic_chain_spiral3d` | Морфологическая роль точки внутри причинной цепочки; `unassigned_resonance` здесь трактуется не как шум, а как пока неразрешённый причинный отклик | Новый главный вывод Блока 4 |
| `unassigned_ratio` | lineage summaries, augmented passports | Доля пока не назначенных точек; скорее мера незавершённости текущей lineage-модели, чем мера “пустого остатка” | Где модель lineage ещё слаба и где могут жить дальние важные отклики |
| Event lifecycle class | `resonance_event_lifecycle_tracker_v2_cli.py`, legacy audits | Новое рождение, продолжение, повторное возбуждение, волна, след | Ансамбль, живое исполнение |
| `LIKELY_INSTRUMENT_BODY_RETURN` | acoustic audits | Возврат корпуса, не равный новой ноте | Разделение note vs body |
| `LIKELY_INTERNAL_WAVE` | acoustic audits | Внутренняя волна той же ноты, а не новое событие | Жизнь sustain |
| `LIKELY_HALL_OR_FIELD_TRACE` | acoustic audits | Дальний полевой/зальный след | Разделение instrument vs space |
| `LIKELY_TRUE_REEXCITATION` | acoustic audits | Реальное новое локальное возбуждение | Событийная дисциплина |
| `dominant_instrument` | layered assignment tools | Кто пока выглядит главным владельцем события | Грубая инструментальная аффинность |
| `support_instruments` | layered assignment tools | Кто присутствует как поддержка, не как единственный владелец | Shared events |
| `attack_owner` | `instrument_role_behavior_mapper.py` | Кто владеет атакой события | Role-aware ensemble model |
| `sustain_owner` | `instrument_role_behavior_mapper.py` | Кто удерживает длительную жизнь события | Strings vs piano vs organ |
| `body_owner` | `instrument_role_behavior_mapper.py` | Кто даёт возврат корпуса | Piano body, instrument body layer |
| `field_owner` | `instrument_role_behavior_mapper.py` | Кто отвечает за полевой / зальный остаток | Hall-aware analysis |
| `support_owner` / `support_owners` | layered and role-aware tools | Кто присутствует как поддерживающий слой | Mixed windows |
| `role_pattern` | `instrument_role_behavior_mapper.py` | Тип ролевой сцены: primary, body return, field trace, shared | Онтология ансамблевого события |
| `role_confidence` | `instrument_role_behavior_mapper.py` | Уверенность в ролевой интерпретации | Фильтрация сильных/слабых решений |
| `shared_mode` / `ownership_mode` | shared-event audits | Тип совместного владения, например piano-attack vs cello-sustain | Разделение совместных событий |
| Passport affinity / ownership score | affinity audits, passport ownership filter | Насколько локальный материал похож на паспорт инструмента | Сопоставление с одиночными паспортами |

---

## 3. Сводка по уровням анализа

Если сжать всё в несколько уровней, то сейчас мы умеем анализировать инструмент по таким этажам:

### Уровень A. Нота как причина

Сюда входят:

- root;
- harmonic chain;
- theoretical agreement;
- tuner confidence.

Это отвечает на вопрос:

> какая нота здесь действительно родилась и насколько она устойчива?

### Уровень B. Нота как резонансная жизнь

Сюда входят:

- amplitudes гармоник;
- note-specific box;
- spiral12 / spiral3d;
- harmonic lineage.

Это отвечает на вопрос:

> как именно нота живёт, что она порождает и как развивается во времени?

### Уровень C. Инструмент как общее тело

Сюда входят:

- box_all;
- breath layer;
- resonance layer;
- box-harmonic relation;
- instrument passport.

Это отвечает на вопрос:

> что у инструмента является повторяющимся собственным резонансным поведением независимо от конкретной ноты?

### Уровень D. Инструмент как участник общего события

Сюда входят:

- lifecycle;
- body return;
- internal wave;
- hall / field trace;
- layered assignment;
- role ownership.

Это отвечает на вопрос:

> какую роль инструмент играет в этом общем музыкальном моменте?

---

## 4. Самый важный вывод

Сейчас мы можем анализировать инструмент уже не только:

- по спектру,
- не только по частоте,
- и не только по усреднённому тембру.

Мы можем анализировать его:

- как причину ноты,
- как структуру гармоник,
- как поведение амплитуд,
- как body-box систему,
- как пространственно-временную геометрию,
- как lineage-порождение вторичных цепочек,
- как область ещё не разрешённых, но потенциально важных дальних откликов,
- и как участника общего события с разными ролями.

Именно поэтому проект постепенно уходит от простого вопроса:

> “какой это инструмент?”

к более точному:

> “какой инструмент что именно делает в этой локальной фазе события?”

---

## 5. Практический итог в числах

На данный момент можно уверенно говорить так:

- **10 крупных классов признаков** уже работают на одиночных тональных инструментах;
- **15 крупных классов**, если включать ансамблевое поведение;
- **30+ конкретных параметров** уже доступны для анализа или напрямую присутствуют в существующих CSV / JSON / MD отчётах.

Следующий важный шаг:

- собрать это в английскую GitHub-версию;
- и затем сделать формальную таблицу `parameter schema` уже в машинно-читаемом виде, чтобы исследователь мог не только читать описание, но и автоматически извлекать эти признаки из данных.
