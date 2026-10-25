# Распознавание нестандартных государственных регистрационных знаков

Полуфинал — «Искусственный интеллект и анализ данных», АИС Город.

Проект реализует офлайн-пайплайн для `type1`, `type1a`, `type1b` и `other`: быстрые region proposals, обученный локальный SVM-фильтр кандидатов, перспективное выравнивание, классификацию типа, обученный локальный OCR символов с детерминированным fallback, проверку маски ГРЗ и формирование CSV.

## Быстрый запуск

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python main.py --input test_images --output results/result.csv --benchmark
```

Формат результата:

```text
image;plate_num;plate_type;confidence
img_0042.jpg;A123BC716;type1a;0.9300
```

## Что уже лежит в models/

- `models/detector_svm.joblib` — обученный HOG+HSV SVM-фильтр кандидатов номера.
- `models/ocr_svm.joblib` — обученный HOG+LinearSVC классификатор допустимых символов.
- `models/ocr_cnn.pt` — экспериментальный CNN bootstrap checkpoint; основной inference использует SVM, поскольку он показал более сильную локальную валидацию в этой сборке.

Фактические метрики сборки находятся в `TRAINING_METRICS.md`. Они измерены на синтетической held-out выборке и не выдаются за качество скрытого теста.

## Датасет

В комплекте 5600 воспроизводимых синтетических изображений:

- `type1`: 1662
- `type1a`: 1729
- `type1b`: 1609
- `other`: 600

Генераторы:

```bash
python dataset/generator/generate.py --count 5000 --seed 20261025 --root dataset
python dataset/generator/generate_other.py --count 600 --seed 20261026 --root dataset
```

## Реальные легальные данные

В `legal_sources/` зафиксированы проверенные источники и лицензии. Для массового сбора из Wikimedia Commons предусмотрен:

```bash
python legal_sources/collect_commons_real.py --root dataset --type1a 150 --type1b 300
```

Скрипт:

1. запрашивает метаданные Commons;
2. принимает только allowlist свободных лицензий;
3. скачивает изображение;
4. размывает найденные лица;
5. использует **наш собственный** детектор для предложения bbox/quad;
6. пишет источник и лицензию;
7. ведёт `dataset/annotation_audit.csv`, где разметка явно обозначена как автоматическая, а не ручная.

Неизвестные символы записываются как `#`, а не выдумываются. После автоматического сбора рекомендуется визуальный аудит `annotation_audit.csv` перед официальной отправкой.

Отдельно: публичные реальные жёлтые номера могут иметь буквенно-цифровой порядок, отличный от конкурсной маски. Формальный датасет и post-processing следуют именно ТЗ полуфинала.

## Дополнительные открытые источники

`legal_sources/sources.csv` фиксирует, в частности:

- CC BY 4.0 OCR-датасет российских номеров на Hugging Face;
- CC BY 4.0 датасет детекции обычных российских номеров;
- CC BY 4.0 набор двухстрочных российских номеров;
- CC0/Public Domain/CC BY-SA эталонные изображения Wikimedia Commons.

## Обучение

Локальные модели можно воспроизвести командами:

```bash
python training/train_ocr_svm.py --per-char 120 --out models/ocr_svm.joblib
python training/train_detector_svm.py --root dataset --limit 1200 --out models/detector_svm.joblib
```

Для опционального YOLO-дообучения:

```bash
pip install -r requirements-train.txt
python training/prepare_yolo.py --dataset dataset --out yolo_dataset
python training/train_detector.py --data yolo_dataset/data.yaml --epochs 80
```

## Валидация

```bash
python official_validator_runner.py
```

Если рядом лежит полученный от организаторов `official_validate_dataset.py`, запускается он. Иначе запускается наш строгий совместимый валидатор. Мы намеренно не называем локальный скрипт «официальным»: в выданном PDF вместо ссылки на официальный файл оставлен placeholder.

Локальный валидатор проверяет структуру, пути, типы, маску, bbox/quad, границы изображения, флаги, source/license, дубликаты и объёмы данных.

## Проверка сборки

```bash
python -m pytest -q
python dataset/validate_dataset.py --root dataset
python main.py --input test_images --output results/result.csv --benchmark
```

Контрольный запуск на локальном ПК участника (25.10.2026): **3 тестовых изображения, 2 детекции, 30.63 мс/изображение**. Это укладывается в требование задания <100 мс/изображение, но не является замером на референсной конфигурации жюри (i5-7600/GTX 1050 Ti).

> Важно: веса SVM сохранены с `scikit-learn 1.8.0`, поэтому версия библиотеки зафиксирована в `requirements.txt`, чтобы исключить несовместимость сериализованных моделей.
