# Tests

72 tests covering data loading, models, training, visualization, end-to-end integration, and the demo entry-point.
With the full project environment installed, the expected result is `72 passed`.
If `gradio` is missing, the 6 demo tests are skipped automatically.
Framework: [pytest](https://docs.pytest.org/) with configuration in `pyproject.toml`.

---

## How to run

```bash
# all tests (verbose output, short tracebacks — configured in pyproject.toml)
pytest

# quick summary
pytest -q

# with coverage report
pytest --cov=data --cov=models --cov=training --cov=visualization
```

All commands should be run from the **project root**.

---

## What's tested

| File | Module | Key checks |
|------|--------|-----------|
| `test_dataset.py` | `data.dataset` | Transforms shape, `AIDataset` length/output, stratified split with no leakage, corrupted image placeholder, CSV validation |
| `test_models.py` | `models.*` | `build_model` dispatch, unknown name raises `ValueError`, forward pass shape `(1,2)`, backbone freeze, selective unfreeze |
| `test_trainer.py` | `training.*` | `seed_everything` determinism, `_compute_metrics` values, one-epoch training loss is finite, NaN loss guard, config validation |
| `test_visualization.py` | `visualization.visualize` | Confusion matrix, ROC curve, and training curve plots save without error |
| `test_integration.py` | end-to-end | Build tiny dataset → train 1 epoch → evaluate → verify outputs |
| `test_demo.py` | `demo.app` | Module imports cleanly, checkpoint directory constants are well-formed `Path` objects, checkpoint auto-discovery works, hard-coded port matches docs |

---

## Fixtures

Shared fixtures live in `conftest.py`:

- **`tmp_image_dir`** — creates a temporary directory with 20 synthetic 64×64 JPEG images (10 per class) for tests that need real files on disk.
- **`MinimalConfig`** — lightweight dataclass mirroring the training config, used to construct `Trainer` without loading a YAML file.
