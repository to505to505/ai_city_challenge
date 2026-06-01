# docs

Документация по обучению на Hafnia Training-aaS для этого репо.

- [metrics-and-checkpoints.md](metrics-and-checkpoints.md) — как ПРАВИЛЬНО логировать метрики и
  чекпоинты в дашборд Hafnia: контракт платформы (`path_model_checkpoints()` / `path_model()`,
  `log_scalar` / `log_metric`), главный грабль с per-thread MLflow run и фиксом (захват + ре-байнд
  run_id), паттерн `TrainStreamingWatcher`, правило rank-0 под DDP, чеклист и таблица симптомов.

См. также [`../trainer_instruction.txt`](../trainer_instruction.txt) — общие факты про платформу
(network isolation, монтирование датасета, сборка trainer.zip, отладка экспериментов).
