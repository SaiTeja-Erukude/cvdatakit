# cvdatakit – Computer Vision Data-Centric Toolkit

A Python library for **dataset-centric CV work**: label-quality checks,
class-imbalance analysis, mislabel detection, and active-learning loop
orchestration. Targets COCO/ImageNet datasets and their long-tail derivatives.

---

## Features

| Module | What it does |
|--------|-------------|
| `cvdatakit.stats` | Dataset statistics: class counts, bbox distributions, Gini/entropy imbalance metrics, co-occurrence matrix |
| `cvdatakit.quality` | Annotation integrity checks (out-of-bounds, duplicates, tiny boxes), Confident-Learning label-quality scoring, kNN-based mislabel detection |
| `cvdatakit.active_learning` | Uncertainty (entropy, margin, LC, BALD), Diversity (CoreSet, cluster-margin, MinMax), Error-Localization (gradient norm, spatial entropy) strategies + loop orchestrator |
| `cvdatakit.recipes` | Ready-made pipelines for COCO and ImageNet-style datasets |
| `cvdatakit.io` | COCO-format reader + HTML/JSON report generator |
| `cvdatakit.cli` | `cvdatakit` CLI: `stats`, `check`, `report`, `imagenet` commands |

---

## Installation

```bash
# Core (no ML framework required)
pip install cvdatakit

# With PyTorch backend
pip install "cvdatakit[torch]"

# With TensorFlow backend
pip install "cvdatakit[tensorflow]"

# Everything + dev tools
pip install "cvdatakit[all,dev]"
```

---

## Quick Start

### Dataset statistics

```python
from cvdatakit.io import COCODataset
from cvdatakit.stats import DatasetStats

ds = COCODataset("annotations/instances_train2017.json")
stats = DatasetStats(ds)
print(stats.summary())
# {'num_images': 118287, 'num_categories': 80, 'class_imbalance': {'gini': 0.42, ...}, ...}

# Long-tail analysis
print(stats.tail_categories(percentile=10))
# ['toaster', 'hair drier', 'parking meter', ...]
```

### Annotation quality checks

```python
from cvdatakit.quality import AnnotationChecker

checker = AnnotationChecker(ds, min_bbox_area=4.0, max_overlap_iou=0.85)
summary = checker.summary()
print(f"Total issues: {summary['total_issues']}")
# {'total_issues': 312, 'by_type': {'out_of_bounds': 5, 'near_duplicate': 307}, ...}
```

### Label quality scoring (Confident Learning)

```python
from cvdatakit.quality import LabelQualityScorer
import numpy as np

# pred_probs: (N, K) out-of-fold predictions from your model
lq = LabelQualityScorer(pred_probs, labels)
issues = lq.ranked_issues(top_k=50)   # worst labels first
print(lq.summary())
# {'estimated_error_rate': 0.032, 'flagged_count': 47, ...}
```

### Mislabel detection

```python
from cvdatakit.quality import MislabelDetector

md = MislabelDetector(embeddings, labels, n_neighbors=15)
candidates = md.rank_candidates(top_k=100)
# [{'index': 2341, 'given_label': 3, 'suggested_label': 7, 'quality_score': 0.12}, ...]
```

### Active learning

```python
from cvdatakit.active_learning import ActiveLearningLoop, UncertaintyStrategy
from cvdatakit.active_learning.backends import PyTorchBackend
from cvdatakit.active_learning.loop import LoopConfig
import torchvision.models as M

model = M.resnet18(weights=M.ResNet18_Weights.DEFAULT)
backend = PyTorchBackend(model, device="cuda")
strategy = UncertaintyStrategy("entropy")

loop = ActiveLearningLoop(
    backend, strategy, images, labels,
    config=LoopConfig(budget_per_round=200, max_rounds=5),
)
history = loop.run()
print(loop.summary())
```

### COCO full-pipeline recipe

```python
from cvdatakit.recipes import COCORecipe

recipe = COCORecipe(
    "annotations/instances_train2017.json",
    image_dir="/data/coco/train2017",
    report_dir="./reports",
    dataset_name="COCO-2017-train",
)
result = recipe.run()
# Writes reports/instances_train2017_report.json + .html
```

---

## CLI

```bash
# Print dataset statistics
cvdatakit stats annotations/instances_val2017.json

# Run annotation checks
cvdatakit check annotations/instances_val2017.json --min-bbox-area 4 --max-iou 0.85

# Generate full HTML + JSON report
cvdatakit report annotations/instances_val2017.json --output-dir ./reports --name "COCO-val"

# Analyse an ImageNet-style folder
cvdatakit imagenet /data/imagenet/val --output-dir ./reports
```

---

## Supported Dataset Formats

### Natively supported (no glue code needed)

| Format | Entry point |
|--------|-------------|
| COCO JSON (`instances_*.json`) | `COCODataset` + `COCORecipe` |
| ImageNet flat-folder (`root/class_name/*.jpg`) | `ImageNetRecipe` |

### Works with any dataset — via numpy arrays

The stats, quality, and active-learning modules are **format-agnostic**. They only need:

| Module | What it needs |
|--------|--------------|
| `LabelQualityScorer` | `(N, K)` pred_probs + `(N,)` labels |
| `MislabelDetector` | `(N, D)` embeddings + `(N,)` labels |
| All 3 AL strategies | numpy arrays (probs / embeddings / gradients) |
| `ActiveLearningLoop` | any image list + numpy labels |

Pascal VOC, Open Images, Roboflow exports, custom CSVs, etc. all work — load your data into numpy arrays or convert to a `COCODataset`.

### What needs a converter

- **Pascal VOC XML / YOLO `.txt`** — no built-in reader; trivial to convert to COCO JSON or use quality/AL modules directly with numpy arrays.
- **Segmentation masks** (`stuff_*.json`, panoptic) — `COCODataset` loads them (still COCO JSON) but `AnnotationChecker` currently only inspects bboxes, not polygon/RLE masks.
- **HuggingFace Datasets / TFRecords / LMDBs** — load to numpy/PIL, pass to AL backends.

### Any format → quality + active learning

```python
# Your own loader — Pascal VOC, YOLO, CSV, anything
embeddings = my_loader.get_embeddings()   # (N, D)
labels      = my_loader.get_labels()      # (N,)
pred_probs  = my_model.predict(images)    # (N, K)

from cvdatakit.quality import LabelQualityScorer, MislabelDetector
from cvdatakit.active_learning.strategies import UncertaintyStrategy

lq       = LabelQualityScorer(pred_probs, labels)
md       = MislabelDetector(embeddings, labels)
strategy = UncertaintyStrategy("entropy")
indices  = strategy.query(pred_probs, budget=100)
```

---

## Project Structure

```
cvdatakit/
├── stats/              Dataset statistics & imbalance metrics
├── quality/            Label quality, mislabel detection, annotation checks
├── active_learning/
│   ├── strategies/     uncertainty / diversity / error-localization
│   ├── backends/       PyTorch, TensorFlow (pluggable)
│   └── loop.py         Loop orchestrator
├── recipes/            COCO & ImageNet pipelines
├── io/                 COCO reader + report generator
└── cli/                Click-based CLI
tests/                  pytest suite (~60 tests)
```

---

## Publishing to PyPI

```bash
pip install build twine
python -m build
twine upload dist/*
```

---

## License

MIT
