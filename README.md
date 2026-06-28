# st-hgrec

Trust-aware sentiment-graph recommendation — offline pipeline.

Setup + how to reproduce every number.

The recommender, the sentiment pipeline, and the graph that together produce
faithful, aspect-level explanations. The paper's contribution is the calibration
study; the offline system only has to be solid and reproducible.

## Two environments (locked)

Two separate Python environments. They never run in the same process, so their
dependencies cannot clash. Pin versions in both from day one.

| Env | Stack | Notes |
| --- | --- | --- |
| Recommender | RecBole + RecBole-GNN | Built-in Popularity, BPR, NGCF, LightGCN |
| Sentiment | PyABSA | Separate env; writes aspect→sentiment to file |

```bash
# recommender env
python -m venv env-rec && source env-rec/bin/activate
pip install -r environments/requirements-rec.txt

# sentiment env (separate)
python -m venv env-absa && source env-absa/bin/activate
pip install -r environments/requirements-absa.txt
```

`src/sentiment` never imports from the recommender side; they meet only through
files in `data/`.

## Dataset

Amazon Reviews 2023, primary category **Video Games**
(`huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023`,
configs `raw_review_Video_Games`, `raw_meta_Video_Games`).
Use the official 5-core benchmark split. `data/` flows raw → interim → processed;
the RecBole atomic files in `processed/` are the only thing the recommender reads.

## Reproducibility

`configs/` is the reproducibility spine — one YAML per run, each carrying its seed
and metric settings, so every table in the paper traces back to a config and a seed.

## Build order

1. Two envs set up and version-pinned
2. Load Video Games; print basic stats
3. Convert 5-core split to RecBole atomic files (`.inter`)
4. Run Popularity + BPR; record Recall@K, NDCG@K
5. PyABSA on ~300 reviews; fix aspect list; eyeball 20
6. Add LightGCN + NGCF to baseline table
7. Build the graph; attach sentiment + trust edges
8. ST-HGRec (LightGCN-heterogeneous over our graph)
9. Explanation generator (honest + ungrounded) + faithfulness check
10. Switch-off (ablation) tests
