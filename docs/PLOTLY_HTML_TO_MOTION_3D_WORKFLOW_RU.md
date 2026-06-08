# Plotly HTML To Motion 3D Workflow RU

## Суть

Apple Motion не импортирует Plotly HTML как настоящую 3D-сцену.

Правильный путь сейчас такой:

`Plotly HTML -> OBJ/MTL sculpture -> USDZ on macOS -> Apple Motion`

## Что уже сделано

В проекте есть конвертер:

- [plotly_html_to_motion_obj_cli.py](/E:/Duodecimal_resonant_numeration/tools/plotly_html_to_motion_obj_cli.py)

Он:

1. читает Plotly HTML, где есть `const traces = [...]`;
2. извлекает `x / y / z / name / marker.color / marker.size`;
3. строит 3D-скульптуру данных как набор маленьких low-poly point-объектов;
4. сохраняет:
   - `OBJ`
   - `MTL`
   - `points CSV`
   - `traces CSV`
   - `scene JSON`
   - `summary TXT`

## Preset-режимы

Сейчас у конвертера есть такие preset-режимы:

- `note_compare_scientific`
- `note_compare_cinematic`
- `note_compare_cinematic_z025`
- `note_compare_cinematic_z035`
- `note_compare_cinematic_z050`
- `note_compare_cinematic_z300`
- `note_compare_cinematic_z500`
- `harmonic_compare_scientific`
- `harmonic_compare_cinematic`

### Для note compare

- `note_compare_scientific`
  - `x = 1`
  - `y = 1`
  - `z = 1`
  - честная геометрия `1:1:1`

- `note_compare_cinematic`
  - `x = 1`
  - `y = 1`
  - `z = 0.12`
  - сильно сплющенная cinematic-версия

- `note_compare_cinematic_z025`
  - `x = 1`
  - `y = 1`
  - `z = 0.25`

- `note_compare_cinematic_z035`
  - `x = 1`
  - `y = 1`
  - `z = 0.35`

- `note_compare_cinematic_z050`
  - `x = 1`
  - `y = 1`
  - `z = 0.50`

- `note_compare_cinematic_z300`
  - `x = 1`
  - `y = 1`
  - `z = 3.00`

- `note_compare_cinematic_z500`
  - `x = 1`
  - `y = 1`
  - `z = 5.00`

Именно три последних режима сейчас нужны для подбора золотой середины между:

- `z = 1.0` -> “резонансный небоскрёб”
- `z = 0.12` -> слишком сильная “тарелка”

Цель:

- сохранить читаемую спираль;
- вернуть ощущение времени;
- выбрать лучший `z-scale` до этапа материалов и цветов.

Отдельно появился и второй класс cinematic-проверки:

- `z300`
- `z500`

Он нужен для обратной задачи:

- не распластать сцену,
- а намеренно вернуть сильное ощущение временной оси,
- если для человека `z050` всё ещё слишком мало как визуальная шкала времени.

## Почему это нужно

`note_compare_3d__*.html` использует:

- `x = x12`
- `y = y12`
- `z = time_sec`

Plotly автоматически подбирает сцену художественно.

`OBJ` экспортирует честные координаты, поэтому без отдельного `z-scale`:

- либо время слишком доминирует;
- либо форма слишком сплющивается.

## Что уже получается

На выходе мы получаем не “график”, а 3D-объект исследования:

- траектории;
- облака;
- объём;
- геометрию, пригодную для облёта камерой в Motion.

## Что ещё не хватает до Apple Motion

На этой Windows-машине пока нет рабочего финального `USDZ` toolchain.

Поэтому до полного Motion-формата не хватает:

1. `USDZ`
2. Mac/Apple toolchain для `OBJ/MTL -> USDZ`
3. финальной проверки в Blender / Quick Look / Reality Composer Pro

То есть сейчас готов промежуточный и уже полезный формат:

- `OBJ`
- `MTL`

А финальный шаг такой:

1. взять `OBJ + MTL`;
2. на Mac конвертировать в `USDZ`;
3. импортировать `USDZ` в Apple Motion.

## Отдельный этап: semantic colors

После выбора правильной геометрии можно включать не plotly-цвета, а
семантическую раскраску trace-слоев.

Для этого в конвертер добавлен:

- `--color-mode note_compare_semantic`

Он раскрашивает `note_compare`-trace по схеме:

- `instrument / chain`
- `instrument / note_box`
- `instrument / dense_other`

То есть цвет начинает показывать не просто точки, а роль слоя внутри ноты.

## Пример запуска

```powershell
python E:\Duodecimal_resonant_numeration\tools\plotly_html_to_motion_obj_cli.py `
  --html E:\Duodecimal_resonant_numeration\Block004_data\_multi_instrument_compare\note_compare_3d__9.A-.html `
  --outdir E:\Duodecimal_resonant_numeration\Block004_data\_multi_instrument_compare\motion_export__note_compare__9.A-__cinematic_z035 `
  --preset note_compare_cinematic_z035 `
  --center-origin `
  --with-axes
```

Или для сильной временной протяжённости:

```powershell
python E:\Duodecimal_resonant_numeration\tools\plotly_html_to_motion_obj_cli.py `
  --html E:\Duodecimal_resonant_numeration\Block004_data\_multi_instrument_compare\note_compare_3d__9.A-.html `
  --outdir E:\Duodecimal_resonant_numeration\Block004_data\_multi_instrument_compare\motion_export__note_compare__9.A-__cinematic_z500 `
  --preset note_compare_cinematic_z500 `
  --center-origin `
  --with-axes
```

## Что появляется в каталоге экспорта

- `plotly_point_sculpture.obj`
- `plotly_point_sculpture.mtl`
- `plotly_points.csv`
- `plotly_traces.csv`
- `plotly_point_sculpture_scene.json`
- `plotly_point_sculpture_summary.txt`

## Важное правило текущего этапа

Цвета и материалы пока не являются главным вопросом.

Сначала надо выбрать правильную геометрию:

- `z025`
- `z035`
- `z050`
- `z300`
- `z500`

И только после этого переходить к:

- цветам trace-слоёв;
- материалам;
- финальному `USDZ`;
- анимации в Motion.
