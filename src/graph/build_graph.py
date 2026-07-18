"""Nodes + typed edges (incl. sentiment) for the ST-HGRec heterogeneous graph.

Builds the heterogeneous graph: user/item/aspect nodes and typed edges,
attaching the aspect->sentiment signals and per-user trust scores. This graph is
what ST-HGRec runs over.

Wiring (P1, fix #3 + test-leakage guard):
  * Interactions come from ``video_games.train.inter`` ONLY. The graph is built
    from TRAIN interactions so that no valid/test edges leak into the structure
    ST-HGRec trains and message-passes over. The combined ``video_games.inter``
    is intentionally no longer read (it merged train+valid+test).
  * Sentiment comes from ``data/interim/absa_normalized.csv`` — the canonical
    aspect-vocabulary output produced by ``run_absa.py`` then
    ``normalize_aspects.py`` (raw open-vocab ``absa_full.csv`` is NOT read here).
  * Trust (fix #4) stays a USER-NODE ATTRIBUTE (``trust_score``) for now. The
    node-attribute vs typed-edge decision is deferred to the ST-HGRec design
    phase (P4); do not convert to edges here.

Sentiment rows reference (user, item) pairs. Rows whose user or item is not in
the TRAIN graph are skipped (they include legitimate test-only pairs); skipped
rows are counted and reported rather than silently dropped.
"""

import argparse
from pathlib import Path
import pickle

import networkx as nx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
# TRAIN interactions only — leakage guard: valid/test edges must not enter the
# graph structure ST-HGRec trains on. The combined video_games.inter is dead.
INTER_PATH = REPO_ROOT / "data" / "processed" / "video_games" / "video_games.train.inter"
# Canonical, aspect-normalized sentiment (run_absa.py -> normalize_aspects.py).
SENTIMENT_PATH = REPO_ROOT / "data" / "interim" / "absa_normalized.csv"
TRUST_PATH = REPO_ROOT / "data" / "interim" / "trust_signals.csv"
GRAPH_OUTPUT = REPO_ROOT / "data" / "processed" / "video_games" / "graph.gpickle"


def load_interactions(inter_path: Path = INTER_PATH) -> pd.DataFrame:
    df = pd.read_csv(inter_path, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]  # strip RecBole :type suffixes
    return df


def load_sentiment(sentiment_path: Path = SENTIMENT_PATH) -> pd.DataFrame:
    if not sentiment_path.exists():
        print(
            f"  WARNING: {sentiment_path} not found, skipping sentiment edges. "
            "Run src/sentiment/run_absa.py then src/sentiment/normalize_aspects.py "
            "to produce absa_normalized.csv."
        )
        return pd.DataFrame(columns=["user_id", "item_id", "aspect", "sentiment"])
    return pd.read_csv(sentiment_path)


def load_trust(trust_path: Path = TRUST_PATH) -> pd.DataFrame:
    if not trust_path.exists():
        print(f"  NOTE: {trust_path} not found — run src/data/trust_signals.py first; skipping trust")
        return pd.DataFrame(columns=["user_id", "trust_score"])
    return pd.read_csv(trust_path)


def build_graph(
    interactions: pd.DataFrame,
    sentiment: pd.DataFrame,
    trust: pd.DataFrame,
) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    users = interactions["user_id"].unique()
    items = interactions["item_id"].unique()
    aspects = sentiment["aspect"].unique() if len(sentiment) else []

    G.add_nodes_from((f"user:{u}" for u in users), node_type="user")
    G.add_nodes_from((f"item:{i}" for i in items), node_type="item")
    G.add_nodes_from((f"aspect:{a}" for a in aspects), node_type="aspect")

    print(f"  added {len(users):,} user nodes, {len(items):,} item nodes, "
          f"{len(aspects):,} aspect nodes")

    # interacts edges: user -> item (vectorized bulk construction).
    interacts_edges = [
        (f"user:{u}", f"item:{i}",
         {"edge_type": "interacts", "rating": r, "timestamp": t})
        for u, i, r, t in zip(
            interactions["user_id"], interactions["item_id"],
            interactions["rating"], interactions["timestamp"],
        )
    ]
    G.add_edges_from(interacts_edges)
    print(f"  added {len(interacts_edges):,} interacts edges")

    # sentiment edges: user -> item (attribute-carrying) AND item -> aspect
    # (typed link). Rows whose user/item is absent from the TRAIN graph are
    # skipped and counted (test-only pairs land here — expected and correct).
    sentiment_edges = []
    has_aspect_edges = []
    skipped_sentiment = 0
    if len(sentiment):
        node_set = G.nodes  # membership test against built node set
        for u, i, a, s in zip(
            sentiment["user_id"], sentiment["item_id"],
            sentiment["aspect"], sentiment["sentiment"],
        ):
            u_node = f"user:{u}"
            i_node = f"item:{i}"
            if u_node not in node_set or i_node not in node_set:
                skipped_sentiment += 1
                continue
            a_node = f"aspect:{a}"
            sentiment_edges.append(
                (u_node, i_node,
                 {"edge_type": "sentiment", "aspect": a, "sentiment": s})
            )
            has_aspect_edges.append(
                (i_node, a_node, {"edge_type": "has_aspect", "sentiment": s})
            )
    G.add_edges_from(sentiment_edges)
    G.add_edges_from(has_aspect_edges)
    print(f"  added {len(sentiment_edges):,} sentiment edges "
          f"(+ {len(has_aspect_edges):,} has_aspect edges)")
    print(f"  skipped {skipped_sentiment:,} sentiment rows whose user/item is not "
          f"in the TRAIN graph (includes test-only pairs — expected)")

    # trust: applied as a USER-NODE ATTRIBUTE (fix #4). Typed-edge conversion is
    # deferred to the ST-HGRec design phase (P4).
    trust_applied = 0
    if len(trust):
        for u, score in zip(trust["user_id"], trust["trust_score"]):
            u_node = f"user:{u}"
            if u_node in G.nodes:
                G.nodes[u_node]["trust_score"] = score
                trust_applied += 1
    print(f"  applied trust_score to {trust_applied:,} user nodes "
          f"({'none yet — run trust_signals.py' if trust_applied == 0 else 'ok'})")

    return G


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inter", type=Path, default=INTER_PATH,
                        help="TRAIN interactions .inter (default: video_games.train.inter)")
    parser.add_argument("--sentiment", type=Path, default=SENTIMENT_PATH,
                        help="Normalized sentiment CSV (default: absa_normalized.csv)")
    parser.add_argument("--trust", type=Path, default=TRUST_PATH,
                        help="Trust signals CSV (default: trust_signals.csv)")
    parser.add_argument("--out", type=Path, default=GRAPH_OUTPUT,
                        help="Output graph pickle (default: graph.gpickle)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading interactions (TRAIN split only)...")
    interactions = load_interactions(args.inter)

    print("Loading sentiment data...")
    sentiment = load_sentiment(args.sentiment)

    print("Loading trust signals...")
    trust = load_trust(args.trust)

    print("Building graph...")
    G = build_graph(interactions, sentiment, trust)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(G, f)
    print(f"\nSaved graph to {args.out}")

    print("\n=== Graph Summary ===")
    print(f"Total nodes: {G.number_of_nodes():,}")
    print(f"Total edges: {G.number_of_edges():,}")

    for ntype in ["user", "item", "aspect"]:
        count = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == ntype)
        print(f"  {ntype} nodes: {count:,}")

    for etype in ["interacts", "sentiment", "has_aspect"]:
        count = sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") == etype)
        print(f"  {etype} edges: {count:,}")

    print("\n=== Sanity check: reload the graph ===")
    with open(args.out, "rb") as f:
        G_reload = pickle.load(f)
    matches = G_reload.number_of_nodes() == G.number_of_nodes()
    print(f"Reloaded graph: {G_reload.number_of_nodes():,} nodes, "
          f"{G_reload.number_of_edges():,} edges — matches original: {matches}")


if __name__ == "__main__":
    main()
