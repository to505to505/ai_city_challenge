# Как ПРАВИЛЬНО логировать метрики и чекпоинты на Hafnia Training-aaS

Авторитетный референс по тому, как трейнеры в этом репо стримят метрики и чекпоинты в дашборд
Hafnia. Выстрадано на `rfdetr_large` — здесь зафиксировано, чтобы не наступать на грабли заново.
Применяется к ОБОИМ трейнерам:

- RF-DETR: [`scripts/train.py`](../scripts/train.py)
- Cascade/ConvNeXt (mmdet 3.x): [`trainer-convnext/scripts/train.py`](../trainer-convnext/scripts/train.py)

Платформенные факты (network isolation, монтирование датасета, пути) — в
[`trainer_instruction.txt`](../trainer_instruction.txt). Здесь — именно про метрики и чекпоинты.

---

## TL;DR (правила, которые нельзя нарушать)

1. **Чекпоинты пиши в `logger.path_model_checkpoints()`** — это `/opt/ml/checkpoints` в облаке,
   платформа их сама собирает. Финальную/лучшую модель — в `logger.path_model()` (`/opt/ml/model`).
2. **Метрики — только через документированный API**: `logger.log_scalar(...)` для всего, что НЕ
   evaluation, и `logger.log_metric(...)` для eval-серий. Никаких прямых `mlflow.*` из своего кода.
3. **Логируй из фонового потока — но СНАЧАЛА перепривяжи MLflow run в этом потоке.** Иначе метрики
   улетят в «orphan run» и в дашборде их не будет. Это главный грабль (см. ниже).
4. **Под DDP логируй только с rank 0** (`is_main_process()`), иначе 4 процесса задублируют метрики и
   будут драться за файлы чекпоинтов.
5. **Стримь LIVE, не в конце.** `train()` блокируется на часы; без живого стрима дашборд пустой, а
   при краше посреди обучения не остаётся ничего.

---

## 1. Контракт платформы

При `HAFNIA_CLOUD=true` (его выставляет платформа) `HafniaLogger` пишет:

| Что | Метод | Путь в облаке | Поведение |
|---|---|---|---|
| Чекпоинты эпох | `logger.path_model_checkpoints()` | `/opt/ml/checkpoints` (`$MDI_CHECKPOINT_DIR`) | авто-собираются, видны в дашборде |
| Финальная/лучшая модель | `logger.path_model()` | `/opt/ml/model` (`$MDI_MODEL_DIR`) | артефакт «Trained model» |
| Артефакты | — | `/opt/ml/output/data` (`$MDI_ARTIFACT_DIR`) | прочие выходы |
| Метрики (не-eval) | `logger.log_scalar(name, value, step)` | MLflow (VPC) | кривые на дашборде |
| Метрики eval | `logger.log_metric(name, value, step)` | MLflow (VPC) | eval-серии |
| Конфиг/гиперпараметры | `logger.log_configuration(dict)` | MLflow params | один раз в начале |

`HafniaLogger.log_scalar`/`log_metric` под капотом вызывают fluent `mlflow.log_metric(...)`, который
пишет в **активный** MLflow run. Это и есть источник главного нюанса.

**Network isolation:** рантайм не имеет outbound в интернет (доступен только VPC-эндпоинт MLflow).
Поэтому `wandb.ai` НЕ работает в облаке — метрики идут в MLflow Hafnia, а не в W&B.

---

## 2. ГЛАВНЫЙ ГРАБЛЬ: MLflow active-run живёт ПО ПОТОКАМ

MLflow хранит стек активных run'ов **per-thread**. `HafniaLogger(...)` вызывает
`mlflow.start_run()` в **главном** потоке. Если потом `log_scalar`/`log_metric` вызвать из **другого**
потока (например, фонового вотчера, который тейлит файл с метриками), этот поток своего активного
run не видит → fluent `mlflow.log_metric` **создаёт новый авто-именованный «orphan» run**, и метрики
оседают в нём, а не в официальном run эксперимента. Симптом: «обучение идёт, а на дашборде метрик нет».

### Фикс (обязательный для любого фонового логирования)

1. В **главном** потоке запомни run_id официального run'а:

   ```python
   def _capture_mlflow_run_id():
       import mlflow
       active = mlflow.active_run()
       return active.info.run_id if active is not None else None
   ```

2. Внутри фонового потока, ДО первого лога, перепривяжи этот run к стеку ЭТОГО потока:

   ```python
   import mlflow
   mlflow.start_run(run_id=self._mlflow_run_id)   # не end_run() — lifecycle владеет главный поток
   ```

После этого `log_scalar`/`log_metric` из вотчера попадают в правильный run.

> Если логируешь из ГЛАВНОГО потока (например, нативный колбэк фреймворка в train-loop) — этой
> проблемы нет, перепривязка не нужна. Она нужна именно когда логирующий код в отдельном thread.

---

## 3. Проверенный паттерн: `TrainStreamingWatcher`

`model.train()` / `runner.train()` блокируются надолго и сами пишут метрики в файл. Поэтому в обоих
трейнерах рядом крутится фоновый поток-вотчер, который:

```
train-loop → пишет файл метрик → [Watcher thread: ре-байнд MLflow run] → log_scalar/log_metric → дашборд
                                                                       └→ лучший чекпоинт → path_model()
```

Обязанности вотчера (см. класс `TrainStreamingWatcher` в обоих `train.py`):

- **тейлит файл метрик** курсором (по числу уже опубликованных строк — идемпотентно);
- маршрутизирует: ключ с префиксом `val/` `validation/` `test/` → `log_metric` (eval-серия), всё
  остальное → `log_scalar`;
- **зеркалит лучший чекпоинт** в `path_model()` при сдвиге mtime (живой артефакт «Trained model»,
  переживает краш/кил);
- финальный `_tick()` в `stop()` — дослать хвост, который не попал в последний интервал.

### Маршрутизация метрик (scalar vs metric)

```python
is_eval = "/" in key and key.split("/", 1)[0] in {"val", "validation", "test"}
fn = logger.log_metric if is_eval else logger.log_scalar
fn(name=key, value=float(value), step=step)
```

То есть **колонки/ключи метрик называй с префиксом**: `train/loss`, `train/lr`, `val/bbox_mAP`.
Префикс — это и есть способ сказать вотчеру «это eval» vs «это train».

### Шаг (ось X)

- train-метрики — по итерациям (`step = global_iter`), throttled (раз в N итераций);
- eval-метрики — раз в эпоху (валидация раз в эпоху — это inherent, не баг: одна точка на эпоху).

---

## 4. Откуда берётся файл метрик (адаптер под фреймворк)

Вотчер одинаковый; различается ТОЛЬКО источник `metrics.csv`:

| Трейнер | Кто пишет `metrics.csv` | Имена лучших чекпоинтов |
|---|---|---|
| RF-DETR | сам трейнер (его CSVLogger, `train_log_on_step=True`) | `checkpoint_best_total/ema/regular.pth` |
| Cascade/ConvNeXt (mmdet 3.x) | `MetricsCsvHook` (мелкий mmengine-хук) | `best_coco*.pth` |

mmdet своего `metrics.csv` не делает, поэтому в convnext-трейнере добавлен `MetricsCsvHook`
([`trainer-convnext/scripts/train.py`](../trainer-convnext/scripts/train.py)):

- `after_train_iter` (раз в `--scalar-interval` итераций) собирает скаляры из
  `runner.message_hub.log_scalars` → строка `{step, train/loss, train/lr, ...}`;
- `after_val_epoch` берёт dict eval-метрик → строка `{step, val/bbox_mAP, ...}`
  (`coco/bbox_mAP` → `val/bbox_mAP`, чтобы вотчер увёл это в `log_metric`);
- пишет CSV **атомарно** (`tmp` + `os.replace`), чтобы вотчер не прочитал недописанный файл.

Важно: и хук, и вотчер активны **только на rank 0** — см. §5.

---

## 5. Multi-GPU (DDP): логируй только с rank 0

- В RF-DETR multi-GPU отдан PyTorch Lightning, поэтому внешний скрипт (а с ним и `HafniaLogger`,
  и вотчер) фактически логирует с одного процесса.
- В mmdet-трейнере `runner.train()` под `--launcher pytorch` запускается на ВСЕХ rank'ах. Поэтому:
  - `MetricsCsvHook` пишет CSV только при `is_main_process()`;
  - `TrainStreamingWatcher` стартует только при `is_main_process()`.
- Иначе: 4 процесса дублируют метрики и конкурируют за `metrics.csv`/чекпоинты.

> Остаточный нюанс для mmdet под torchrun: сам `HafniaLogger(...)` и экспорт датасета выполняются на
> каждом rank (скрипт стартует per-rank). Для 1 GPU (Lite) неактуально; для Scale экспорт/инициализацию
> логгера стоит дополнительно завести на rank 0 с барьером.

---

## 6. Чеклист перед заливкой

- [ ] Чекпоинты сохраняются в `logger.path_model_checkpoints()` (а не в произвольный `output/`).
- [ ] Лучший чекпоинт зеркалится в `logger.path_model()`.
- [ ] Метрики только через `log_scalar`/`log_metric`; eval-ключи с префиксом `val/`.
- [ ] Если логируешь из фонового потока — `_capture_mlflow_run_id()` в главном + `start_run(run_id=...)`
      в потоке.
- [ ] Стрим LIVE (вотчер стартует ДО `train()`, `stop()` в `finally`).
- [ ] Под DDP — лог и запись файла только при `is_main_process()`.
- [ ] В облаке НЕ передавать `--wandb` (network isolation; wandb.ai недоступен).
- [ ] `mlflow=True` НЕ передавать фреймворку, если HafniaLogger уже владеет run'ом (иначе ВТОРОЙ run).

---

## 7. Типичные симптомы и причины

| Симптом | Причина | Фикс |
|---|---|---|
| Обучение идёт, метрик на дашборде нет | лог из чужого потока → orphan run | ре-байнд run_id в потоке (§2) |
| Появился лишний пустой run рядом | фреймворку передали `mlflow=True` поверх HafniaLogger | не передавать; стримить через вотчер |
| Метрики задублированы ×N | логирование на всех rank DDP | `is_main_process()` гейт (§5) |
| Eval-метрика попала в scalar-кривую | ключ без префикса `val/` | называть `val/<metric>` |
| «Trained model» пустой после краша | зеркалили только в конце | зеркалить по mtime в каждом тике вотчера |
| Дашборд пуст до самого конца | публикация только после `train()` | LIVE-вотчер + `train_log_on_step=True` (RF-DETR) |

---

## 8. Ссылки на код

- RF-DETR: `TrainStreamingWatcher`, `_capture_mlflow_run_id`, `_bind_mlflow_run_in_thread`,
  `_publish_row` — [`scripts/train.py`](../scripts/train.py).
- mmdet 3.x: те же классы + `MetricsCsvHook` —
  [`trainer-convnext/scripts/train.py`](../trainer-convnext/scripts/train.py).
- Платформенные пути, network isolation, отладка экспериментов —
  [`trainer_instruction.txt`](../trainer_instruction.txt).
