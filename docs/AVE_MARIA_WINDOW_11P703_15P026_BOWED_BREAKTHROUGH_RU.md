# Ave Maria Window 11.703-15.026 Bowed Breakthrough RU

## Зачем нужен этот документ

Этот документ фиксирует **первый действительно удачный прорыв** в извлечении
длительного смычкового слоя из окна `11.703–15.026` в `Ave Maria`.

Ключевой результат:

- файл
  [window_bowed_source_locked_stft_v2.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_source_locked_stft_v2.wav)
  дал **почти естественное звучание с реальной амплитудой**, без рояльной
  грязи и без осцилляторной искусственности предыдущих веток;
- на слух пока ещё не хватает микросдвиговой жизни самих гармоник, но по
  качеству это первый слой, который звучит как **реальная запись**, а не как
  синтетический рендер.

Этот момент нужно считать опорной точкой для дальнейшего развития `Block001`.

## Контекст окна

Рабочее окно:

- `11.703–15.026` сек

Каталог окна:

- [window_11p703s_15p026s_backbone_second_audit_v1](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1)

Исходный аудио-фрагмент:

- [window_raw_excerpt.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_raw_excerpt.wav)

Главная задача этого окна:

- извлечь как можно больше данных второго длительного bowed-слоя;
- отделить его от рояля;
- не угадывать инструмент раньше времени;
- не усреднять микросдвиги;
- не подменять реальные данные вероятностными добавками;
- в рендере использовать физику исходного сигнала, а не additive-осциллятор.

## Законы, которые привели к результату

Этот результат был достигнут только после того, как были приняты следующие
жёсткие правила:

1. Микросдвиги нельзя усреднять никогда.
2. `coarse_note` не может быть ранней базовой сущностью.
3. Нота и инструмент должны быть разделены.
4. Измеряемое и выводимое должны быть разделены.
5. Паспорт инструмента — поздняя проверка, а не ранний генератор гипотезы.
6. Если рендер звучит как осциллятор, значит в нём потеряна физика исходного
   сигнала, даже если частоты и амплитуды взяты из реальных данных.

Опорные документы:

- [BLOCK001_MICROSHIFT_LAW_RU.md](/E:/Duodecimal_resonant_numeration/docs/BLOCK001_MICROSHIFT_LAW_RU.md)
- [BLOCK001_EVENT_SLOT_ANALYSIS_ORDER_RU.md](/E:/Duodecimal_resonant_numeration/docs/BLOCK001_EVENT_SLOT_ANALYSIS_ORDER_RU.md)
- [BLOCK001_INSTRUMENT_ANALYSIS_PROPERTY_LIST_RU.md](/E:/Duodecimal_resonant_numeration/docs/BLOCK001_INSTRUMENT_ANALYSIS_PROPERTY_LIST_RU.md)

## Хронология пути к прорыву

### Этап 1. Временные маски и ранние role previews

Первые локальные preview строились как activity/time masks:

- [window_main_backbone_preview.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_main_backbone_preview.wav)
- [window_second_sustain_preview.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_second_sustain_preview.wav)

Что стало ясно:

- рояль и второй длительный слой по времени уже различались;
- но time-mask неизбежно таскал рояль внутрь bowed-preview, если они жили в
  одном кадре;
- это было полезно как роль-сцена, но не как реальная звуковая реконструкция.

### Этап 2. Sinebank и harmonicized preview

Дальше пошла ветка `probe-sinebank` и `harmonicized`:

- [window_second_sustain_probe_sinebank_preview_v1.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_second_sustain_probe_sinebank_preview_v1.wav)
- [window_second_sustain_probe_harmonicized_preview_v1.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_second_sustain_probe_harmonicized_preview_v1.wav)
- [window_second_sustain_probe_harmonicized_preview_v2.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_second_sustain_probe_harmonicized_preview_v2.wav)

Полезный вывод:

- `v2` был лучшей из ранних слуховых версий;
- он уже был похож на чистый bowed-тон;
- но оставался выше реального аудио и всё ещё был синтетическим.

Важно:

- именно здесь впервые стало слышно, что второй слой реально существует как
  отдельная жизнь;
- но физика исходного звука ещё не была восстановлена.

### Этап 3. Ошибка синтетического suboctave

Была сделана ветка с искусственным субоктавным добавлением:

- [window_second_sustain_probe_harmonicized_preview_v3_suboctave.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_second_sustain_probe_harmonicized_preview_v3_suboctave.wav)

Это было признано ошибкой:

- добавление нижней опоры синтетически искажало звук;
- правильный принцип: искать опору в самих данных, а не переносить верхний
  смычковый слой вниз автоматически.

### Этап 4. Hidden root / support как data-grounded слой

Дальше была найдена реальная связанная нижняя опора:

- [window_hidden_root_support_summary_v1.txt](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_hidden_root_support_summary_v1.txt)
- [window_hidden_root_support_groups_v1.csv](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_hidden_root_support_groups_v1.csv)
- [window_second_sustain_data_grounded_owner_v1.csv](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_second_sustain_data_grounded_owner_v1.csv)

Ключевое понимание:

- верхний слой `B.5` нельзя считать корнем;
- его надо рассматривать как сильную bowed-chain;
- а root/support надо искать под ним как hidden или shared фундаментальный
  слой.

Это был большой методический шаг, но ещё не прорыв в рендере.

### Этап 5. Паспортная проверка bowed-note структуры

Появились note-passport проверки:

- [window_bowed_note_passport_compare_summary_v1.txt](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_note_passport_compare_summary_v1.txt)
- [window_harmonic_phase_passport_compare_summary_v1.txt](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_harmonic_phase_passport_compare_summary_v1.txt)

Они показали:

- спор `violin/cello` ещё рано решать окончательно;
- но само bowed-note сравнение полезно для логики общего алгоритма;
- особенно важно было понять, какие гармоники и в каком порядке вообще
  проявляются.

### Этап 6. Raw harmonic probe scan

Это был один из главных переломов:

- [window_raw_harmonic_probe_scan_summary_v1.txt](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_raw_harmonic_probe_scan_summary_v1.txt)
- [window_raw_harmonic_probe_scan_summary_v1.csv](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_raw_harmonic_probe_scan_summary_v1.csv)

Что было открыто:

- в raw-данных живут не только `h1/h2`, но и `h3/h5/h7`;
- на выбранных кадрах второго слоя:
  - `h1 coverage = 1.0`
  - `h2 coverage = 1.0`
  - `h3 coverage = 1.0`
  - `h5 coverage = 1.0`
  - `h7 coverage = 0.7826`
- прежние owner-слои были слишком узкими и просто отрезали живую harmonic
  связность ноты.

Это дало ключевой сдвиг:

- данных в bowed-слое гораздо больше, чем казалось по ранним winner/owner
  слоям;
- значит проблема была в логике отбора и связности, а не в отсутствии
  информации.

### Этап 7. Bowed bundle и рост harmonic ceiling

После raw-scan возникла ветка bowed-bundle:

- [window_bowed_bundle_preview_v1.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_bundle_preview_v1.wav)
- [window_bowed_bundle_preview_v2.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_bundle_preview_v2.wav)
- [window_bowed_bundle_preview_v3_highband.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_bundle_preview_v3_highband.wav)
- [window_bowed_bundle_preview_v4_real_amp.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_bundle_preview_v4_real_amp.wav)
- [window_bowed_bundle_preview_v5_spectral_calibrated.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_bundle_preview_v5_spectral_calibrated.wav)

Что дала эта ветка:

- стало ясно, что потолок `h7` слишком низкий;
- пришлось поднимать верх до `h25`, примерно до `16.6 kHz`;
- стало ясно, что амплитуды между гармониками нельзя брать только из
  probe-space: их нужно калибровать по реальному спектру исходного WAV.

Но:

- даже калиброванный additive render оставался синтетическим, потому что
  сохранял структуру частот, но не физику исходного сигнала.

### Этап 8. Microcloud

Следующий важный шаг:

- [window_bowed_bundle_preview_v6_microcloud.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_bundle_preview_v6_microcloud.wav)

Почему он был ценен:

- это уже не одна идеальная линия на гармонику;
- это реальное `micro-cloud` соседних family;
- звук стал живее.

Почему он не стал финалом:

- появился `wow-wow`;
- оставались рваность и дыры;
- стало ясно, что рвётся не одна гармоника, а весь bowed-organism целиком.

### Этап 9. Ошибочные ветки `v7` и `v8`

Потом были сделаны:

- [window_bowed_bundle_preview_v7_fullwindow.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_bundle_preview_v7_fullwindow.wav)
- [window_bowed_bundle_preview_v8_gaprecover.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_bundle_preview_v8_gaprecover.wav)

Обе ветки были признаны неправильными.

Причина зафиксирована здесь:

- [window_gap_diagnostic_v1.txt](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_gap_diagnostic_v1.txt)

Ключевой вывод:

- разрывы были синхронны почти по всем `h1..h25`;
- значит пропадали не отдельные partials;
- сама continuity / ownership логика выключала весь bowed-organism.

Это был один из самых важных отрицательных результатов:

- стало ясно, что нельзя продолжать латать звук одним рендером;
- надо чинить локальную связность family-to-family.

### Этап 10. Local family continuity graph

После этого был построен граф локальной связности:

- [window_local_family_continuity_summary_v1.txt](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_local_family_continuity_summary_v1.txt)
- [window_local_family_continuity_nodes_v1.csv](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_local_family_continuity_nodes_v1.csv)
- [window_local_family_continuity_edges_v1.csv](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_local_family_continuity_edges_v1.csv)
- [window_local_family_gap_candidates_v1.csv](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_local_family_gap_candidates_v1.csv)

Критическое открытие:

- самый большой провал `837..866` имел `107` bowed family-кандидатов внутри;
- bowed-слой в данных присутствовал;
- его выключала именно логика gating.

### Этап 11. Continuity resolver

Это был структурный прорыв:

- [window_bowed_continuity_summary_v1.txt](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_continuity_summary_v1.txt)
- [window_bowed_continuity_resolved_cloud_v1.csv](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_continuity_resolved_cloud_v1.csv)
- [window_bowed_continuity_family_paths_v1.csv](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_continuity_family_paths_v1.csv)
- [window_bowed_continuity_frame_support_v1.csv](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_continuity_frame_support_v1.csv)

Главные числа:

- `selected_harmonic_count = 25`
- `preliminary_active_frames = 201`
- `resolved_active_frames = 201`
- `resolved_union_segment_count = 1`
- единый сегмент: `702..902`
- `resolved_cloud_row_count = 4847`
- `observed_rows = 4759`
- `held_rows = 88`

Что это значило:

- bowed-organism впервые собрался в **одну непрерывную жизнь** на уровне
  структуры;
- это доказало, что проблема рваности была в логике связности, а не в
  отсутствии bowed-данных.

### Этап 12. Strict observed raw additive render

После этого был сделан строгий additive-рендер:

- [window_bowed_strict_observed_raw_v1.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_strict_observed_raw_v1.wav)

Его смысл:

- использовать только `observed`;
- только реальные `frequency_hz`;
- только реальные `raw_energy`;
- никаких `held`, priors, passports и synthetic additions.

Почему он оказался тупиком:

- он звучал как “осциллографическая чушь”;
- значит даже реальных частот и амплитуд недостаточно;
- additive-банк всё равно терял реальную фазу, bow-noise, корпусную остаточность
  и живую STFT-физику исходного сигнала.

Это был последний важный отрицательный шаг перед прорывом.

### Этап 13. Source-locked STFT v1

Следующий поворот был принципиально правильным:

- [window_bowed_source_locked_stft_v1.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_source_locked_stft_v1.wav)
- [window_bowed_source_locked_stft_v1_summary.txt](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_source_locked_stft_v1_summary.txt)
- [window_bowed_source_locked_stft_v1_mask.csv](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_source_locked_stft_v1_mask.csv)

Идея `v1`:

- не синтезировать новые осцилляторы;
- брать сам `window_raw_excerpt.wav`;
- считать STFT;
- оставлять только те bins, которые поддержаны bowed observed rows;
- сохранять **реальную complex phase** исходного сигнала;
- собирать обратно через `iSTFT`.

Это уже был правильный физический класс рендера.

Почему `v1` оказался почти неслышим:

- один ложный пик около конца окна забил всю нормализацию;
- полезный bowed-сигнал был в файле, но почти весь оказался слишком тихим.

### Этап 14. Source-locked STFT v2 — момент прорыва

Именно здесь произошёл прорыв:

- [window_bowed_source_locked_stft_v2.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_source_locked_stft_v2.wav)
- [window_bowed_source_locked_stft_v2_summary.txt](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_source_locked_stft_v2_summary.txt)
- [window_bowed_source_locked_stft_v2_mask.csv](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_source_locked_stft_v2_mask.csv)

Время создания:

- `30.05.2026 23:14:59`

Что было исправлено по сравнению с `v1`:

1. Правильный tail-padding для STFT.
2. Робастная нормализация по percentile вместо опоры на единичный ложный пик.

Технически это было реализовано в:

- [window_bowed_source_locked_stft_reconstructor_cli.py](/E:/Duodecimal_resonant_numeration/py/music12/blocks/Block002_pipeline/window_bowed_source_locked_stft_reconstructor_cli.py)

Ключевые параметры `v2`:

- `fft_size = 4096`
- `hop_size = 735`
- `gaussian_bin_sigma = 2.5`
- `skirt_bin_sigma = 6.0`
- `skirt_gain = 0.28`
- `normalization_percentile = 99.9`

Ключевые числа `v2`:

- `observed_row_count = 4759`
- `active_stft_frame_count = 195`
- `padded_signal_len = 146686`
- `robust_peak_after_render = 0.053143`
- `peak_after_render = 0.950000`

Что изменилось на уровне фактического сигнала:

- bowed-слой перестал быть почти неслышимым;
- появилась **реальная амплитуда**, визуально и на слух близкая к исходному
  характеру;
- исчезла осцилляторная искусственность additive-веток;
- появился **почти естественный** звучащий bowed-слой, отделённый от рояля.

Почему это сработало:

- continuity была уже разрешена на уровне family-to-family структуры;
- рендер больше не выдумывал фазу, а брал её из исходного окна;
- амплитуда перестала подчиняться одному ложному граничному пику;
- bowed ownership стал управлять не синтезом, а **отбором исходных STFT-блоков**.

## Что именно нельзя забыть

Этот результат возник не от одной “хитрой формулы”, а от правильной
последовательности решений:

1. Сначала нужно было доказать, что bowed-данные реально существуют в окне.
2. Потом нужно было доказать, что проблема разрывов сидит в continuity/gating,
   а не в отсутствии harmonics.
3. Потом нужно было починить связность bowed-organism как структуры.
4. И только после этого можно было переходить к source-locked STFT
   reconstruction.

Если нарушить этот порядок, система снова скатывается либо в:

- дырявый time-mask,
- либо в additive oscillator render,
- либо в synthetic harmonic guessing,
- либо в плохую глобальную нормализацию.

## Что всё ещё не хватает

Хотя `v2` уже очень близок к живому звучанию, в нём всё ещё не хватает:

- более богатых микросдвигов самих гармоник;
- более живой внутренней harmonic width;
- мелкой натуральной неустойчивости bowed peaks;
- части той “дыхательной” внутренней жизни, которая видна на живом спектре.

То есть текущий статус:

- **реальная амплитуда** уже почти достигнута;
- **реальная фазовая природа** уже появилась;
- **естественное отделение от рояля** уже очень хорошее;
- но **микросдвиговая жизнь harmonic peaks** ещё не вытащена полностью.

## Что считать главным рабочим итогом

На данный момент для окна `11.703–15.026` главным рабочим результатом нужно
считать именно:

- [window_bowed_source_locked_stft_v2.wav](/E:/Duodecimal_resonant_numeration/Block001_data/Ave_Maria/11_reports_Ave_Maria_clean_probe_rerun_v1/window_11p703s_15p026s_backbone_second_audit_v1/window_bowed_source_locked_stft_v2.wav)

А главным техническим законом, приведшим к этому результату, нужно считать:

> если нужно сохранить естественность bowed-слоя, нельзя останавливаться на
> additive harmonic rendering;
> bowed ownership должен управлять source-locked reconstruction из реального
> STFT исходного окна после предварительного family-level continuity resolution.

## Следующий правильный шаг после этого прорыва

Не менять больше класс рендера обратно на additive.

Следующий шаг должен быть внутри уже правильной ветки:

1. сохранить `source-locked STFT` как базовый класс реконструкции;
2. добавить в него более точную microshift-aware harmonic width;
3. вытащить из данных не только присутствие гармоник, но и их локальную
   внутреннюю микродвижущуюся форму.

Именно это, вероятнее всего, даст следующий прирост качества после `v2`.
