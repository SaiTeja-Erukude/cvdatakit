"""
smoke_test.py – quick manual check of every cvdatakit module.
Run: python smoke_test.py
No real images or annotation files required.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

PASS = "[PASS]"
FAIL = "[FAIL]"

# ── helpers ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def make_coco_json(tmp: Path) -> Path:
    rng = np.random.default_rng(0)
    N_IMG, N_CAT, ANN_PER = 30, 6, 4
    W, H = 640, 480
    cats = [{"id": i, "name": f"class_{i}", "supercategory": ""} for i in range(N_CAT)]
    imgs = [{"id": i, "file_name": f"img_{i}.jpg", "width": W, "height": H} for i in range(N_IMG)]
    anns = []
    ann_id = 0
    for img_id in range(N_IMG):
        for _ in range(ANN_PER):
            x, y = int(rng.integers(0, 500)), int(rng.integers(0, 400))
            w, h = int(rng.integers(20, 100)), int(rng.integers(20, 80))
            anns.append({
                "id": ann_id, "image_id": img_id,
                "category_id": int(rng.integers(0, N_CAT)),
                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
            })
            ann_id += 1
    p = tmp / "annotations.json"
    p.write_text(json.dumps({"images": imgs, "categories": cats, "annotations": anns}))
    return p


# ── 1. io ─────────────────────────────────────────────────────────────────────

section("1. cvdatakit.io  —  COCODataset + ReportGenerator")

with tempfile.TemporaryDirectory() as _tmp:
    tmp = Path(_tmp)
    ann_file = make_coco_json(tmp)

    from cvdatakit.io import COCODataset, ReportGenerator

    ds = COCODataset(ann_file)
    assert ds.num_images == 30
    assert ds.num_categories == 6
    assert ds.num_annotations == 120
    print(f"{PASS}  COCODataset  images={ds.num_images}  cats={ds.num_categories}  anns={ds.num_annotations}")

    train, val = ds.split(train_ratio=0.8)
    assert train.num_images + val.num_images == 30
    print(f"{PASS}  split()  train={train.num_images}  val={val.num_images}")

    rg = ReportGenerator("smoke_test")
    rg.add_section("dummy", {"a": 1, "b": [1, 2, 3]})
    jp = rg.save_json(tmp / "report.json")
    hp = rg.save_html(tmp / "report.html")
    assert jp.exists() and hp.exists()
    print(f"{PASS}  ReportGenerator  json={jp.name}  html={hp.name}")

    # Keep ds alive outside the block
    ann_file2 = Path(tempfile.mktemp(suffix=".json"))
    ann_file2.write_text(ann_file.read_text())

# re-load outside tempdir
ds = COCODataset(ann_file2)


# ── 2. stats ──────────────────────────────────────────────────────────────────

section("2. cvdatakit.stats  —  DatasetStats")

from cvdatakit.stats import DatasetStats

stats = DatasetStats(ds)
summary = stats.summary()

print(f"{PASS}  summary keys: {list(summary.keys())}")

imb = summary["class_imbalance"]
print(f"{PASS}  imbalance  gini={imb['gini']:.4f}  ratio={imb['imbalance_ratio']:.2f}x  eff_classes={imb['effective_num_classes']:.2f}")

bbox = summary["bbox_statistics"]
print(f"{PASS}  bbox stats  total={bbox['total_boxes']}  mean_area={bbox['area']['mean']:.1f}")

tail = stats.tail_categories(percentile=30.0)
print(f"{PASS}  tail_categories (30th pct): {tail}")

mat, names = stats.co_occurrence_matrix()
print(f"{PASS}  co_occurrence_matrix  shape={mat.shape}  classes={names}")

dist = stats.class_distribution()
print(f"{PASS}  class_distribution  top={dist[0]['name']}({dist[0]['count']})  bot={dist[-1]['name']}({dist[-1]['count']})")


# ── 3. quality ────────────────────────────────────────────────────────────────

section("3. cvdatakit.quality  —  AnnotationChecker")

from cvdatakit.quality import AnnotationChecker

checker = AnnotationChecker(ds, min_bbox_area=4.0, max_overlap_iou=0.85)
chk_summary = checker.summary()
print(f"{PASS}  AnnotationChecker  total_issues={chk_summary['total_issues']}")
print(f"       by_type: {chk_summary['by_type']}")
print(f"       by_severity: {chk_summary['by_severity']}")

section("3b. cvdatakit.quality  —  LabelQualityScorer")

from cvdatakit.quality import LabelQualityScorer

rng = np.random.default_rng(42)
N, K = 200, 6
pred_probs = rng.dirichlet(np.ones(K), size=N).astype(np.float32)
labels = rng.integers(0, K, size=N).astype(np.int64)

lq = LabelQualityScorer(pred_probs, labels)
lq_summary = lq.summary()
print(f"{PASS}  LabelQualityScorer  flagged={lq_summary['flagged_count']}/{N}  est_error={lq_summary['estimated_error_rate']:.4f}")

issues = lq.ranked_issues(top_k=5)
print(f"{PASS}  ranked_issues top-5: {[(i['index'], i['given_label'], i['predicted_label']) for i in issues]}")

joint, marginal = lq.confusion_matrix()
print(f"{PASS}  confusion_matrix  shape={joint.shape}  marginal_sum={marginal.sum():.4f}")

section("3c. cvdatakit.quality  —  MislabelDetector")

from cvdatakit.quality import MislabelDetector

k, n_per = 6, 40
centers = np.eye(k, 32)
embeddings = np.vstack([centers[c] + 0.3 * rng.standard_normal((n_per, 32)) for c in range(k)]).astype(np.float32)
emb_labels = np.repeat(np.arange(k), n_per).astype(np.int64)

md = MislabelDetector(embeddings, emb_labels, n_neighbors=8)
knn_scores = md.knn_label_quality()
print(f"{PASS}  knn_label_quality  mean={knn_scores.mean():.3f}  min={knn_scores.min():.3f}")

candidates = md.rank_candidates(top_k=5)
print(f"{PASS}  rank_candidates top-5 (index, given, suggested):")
for c in candidates:
    print(f"       [{c['index']:3d}]  given={c['given_label']}  suggested={c['suggested_label']}  score={c['quality_score']:.3f}")


# ── 4. active_learning.strategies ────────────────────────────────────────────

section("4a. active_learning  —  UncertaintyStrategy")

from cvdatakit.active_learning.strategies import UncertaintyStrategy

pool_probs = rng.dirichlet(np.ones(K), size=300).astype(np.float32)

for method in ("entropy", "margin", "least_confidence"):
    strat = UncertaintyStrategy(method)
    idx = strat.query(pool_probs, budget=30)
    assert len(idx) == 30
    print(f"{PASS}  {method:20s}  queried={len(idx)}  top_idx={idx[0]}")

# BALD with MC samples
mc = rng.dirichlet(np.ones(K), size=(10, 300)).astype(np.float32)
strat_bald = UncertaintyStrategy("bald")
scores_bald = strat_bald.score(pool_probs, mc_samples=mc)
print(f"{PASS}  bald (mc_samples)  scores shape={scores_bald.shape}  mean={scores_bald.mean():.4f}")

section("4b. active_learning  —  DiversityStrategy")

from cvdatakit.active_learning.strategies import DiversityStrategy

pool_emb = rng.standard_normal((300, 64)).astype(np.float32)
labeled_emb = rng.standard_normal((50, 64)).astype(np.float32)

for method in ("coreset", "cluster_margin", "minmax"):
    strat = DiversityStrategy(method)
    idx = strat.query(pool_emb, budget=30, labeled_embeddings=labeled_emb, pred_probs=pool_probs)
    assert len(idx) == 30
    print(f"{PASS}  {method:20s}  queried={len(idx)}  top_idx={idx[0]}")

section("4c. active_learning  —  ErrorLocalizationStrategy")

from cvdatakit.active_learning.strategies import ErrorLocalizationStrategy

grads = rng.standard_normal((300, 256)).astype(np.float32)
spatial_logits = rng.standard_normal((20, 8, 8, K)).astype(np.float32)
train_losses = rng.random(300).astype(np.float32)

for method, kwargs in [
    ("gradient_norm",    {"gradients": grads}),
    ("spatial_entropy",  {"spatial_logits": spatial_logits}),
    ("influence_approx", {"train_losses": train_losses}),
]:
    strat = ErrorLocalizationStrategy(method)
    scores = strat.score(**kwargs)
    budget = min(30, len(scores))
    idx = strat.query(budget, **kwargs)
    print(f"{PASS}  {method:20s}  scores={scores.shape}  queried={len(idx)}")


# ── 5. recipes ────────────────────────────────────────────────────────────────

section("5a. recipes  —  COCORecipe")

from cvdatakit.recipes import COCORecipe

with tempfile.TemporaryDirectory() as _tmp:
    tmp = Path(_tmp)
    ann = make_coco_json(tmp)
    recipe = COCORecipe(ann, report_dir=tmp, dataset_name="smoke_coco")
    result = recipe.run(save_report=True)
    assert Path(result["report_json"]).exists()
    assert Path(result["report_html"]).exists()
    print(f"{PASS}  COCORecipe.run()  json={Path(result['report_json']).name}  html={Path(result['report_html']).name}")

    weights = recipe.recommend_resampling_weights()
    print(f"{PASS}  resampling weights: { {k: round(v,3) for k,v in list(weights.items())[:3]} } ...")

    lt = recipe.long_tail_report()
    print(f"{PASS}  long_tail_report  tail={lt['tail_count']}  head={lt['head_count']}")

section("5b. recipes  —  ImageNetRecipe")

from cvdatakit.recipes import ImageNetRecipe

with tempfile.TemporaryDirectory() as _tmp:
    tmp = Path(_tmp)
    # create a fake flat-folder structure
    for cls in ["dog", "cat", "bird"]:
        cls_dir = tmp / cls
        cls_dir.mkdir()
        for i in range(5):
            (cls_dir / f"img_{i}.jpg").write_bytes(b"fake")  # dummy files

    recipe = ImageNetRecipe(tmp, report_dir=tmp, dataset_name="smoke_imagenet")
    result = recipe.run(save_report=True)
    assert Path(result["report_html"]).exists()
    imb = recipe.class_imbalance_summary()
    over = recipe.recommend_oversampling()
    print(f"{PASS}  ImageNetRecipe  classes={result['stats']['num_categories']}  images={result['stats']['num_images']}")
    print(f"{PASS}  recommend_oversampling: {over}")


# ── done ──────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("  All smoke tests passed.")
print('='*60)
