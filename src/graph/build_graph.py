"""Nodes + typed edges (incl. sentiment).

Builds the heterogeneous graph: user/item/aspect nodes and typed edges,
attaching the aspect→sentiment signals and the trust edges. This graph is what
ST-HGRec runs over.
"""

from pathlib import Path
import pickle
import networkx as nx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
INTER_PATH = REPO_ROOT / "data" / "processed" / "video_games" / "video_games.inter"
SENTIMENT_PATH = REPO_ROOT / "data" / "interim" / "absa_sample.csv"
TRUST_PATH = REPO_ROOT / "data" / "interim" / "trust_signals.csv"
GRAPH_OUTPUT = REPO_ROOT / "data" / "processed" / "video_games" / "graph.gpickle"


def load_interactions() -> pd.DataFrame:
    df = pd.read_csv(INTER_PATH, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]  # strip RecBole :type suffixes
    return df


def load_sentiment() -> pd.DataFrame:
    if not SENTIMENT_PATH.exists():
        print(f"  WARNING: {SENTIMENT_PATH} not found, skipping sentiment edges")
        return pd.DataFrame(columns=["user_id", "item_id", "aspect", "sentiment"])
    return pd.read_csv(SENTIMENT_PATH)


def load_trust() -> pd.DataFrame:
    if not TRUST_PATH.exists():
        print(f"  NOTE: {TRUST_PATH} not found yet (Day 6 task) — skipping trust edges")
        return pd.DataFrame(columns=["user_id", "trust_score"])
    return pd.read_csv(TRUST_PATH)


def build_graph(
    interactions: pd.DataFrame,
    sentiment: pd.DataFrame,
    trust: pd.DataFrame,
) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    users = interactions["user_id"].unique()
    items = interactions["item_id"].unique()
    aspects = sentiment["aspect"].unique() if len(sentiment) else []

    for u in users:
        G.add_node(f"user:{u}", node_type="user")
    for i in items:
        G.add_node(f"item:{i}", node_type="item")
    for a in aspects:
        G.add_node(f"aspect:{a}", node_type="aspect")

    print(f"  added {len(users):,} user nodes, {len(items):,} item nodes, "
          f"{len(aspects):,} aspect nodes")

    # interacts edges: user -> item
    for _, row in interactions.iterrows():
        G.add_edge(
            f"user:{row['user_id']}",
            f"item:{row['item_id']}",
            edge_type="interacts",
            rating=row["rating"],
            timestamp=row["timestamp"],
        )
    print(f"  added {len(interactions):,} interacts edges")

    # sentiment edges: user -> item (attribute-carrying) AND item -> aspect (typed link)
    sentiment_edge_count = 0
    for _, row in sentiment.iterrows():
        u_node = f"user:{row['user_id']}"
        i_node = f"item:{row['item_id']}"
        a_node = f"aspect:{row['aspect']}"
        if u_node not in G.nodes or i_node not in G.nodes:
            continue
        G.add_edge(
            u_node, i_node,
            edge_type="sentiment",
            aspect=row["aspect"],
            sentiment=row["sentiment"],
        )
        G.add_edge(
            i_node, a_node,
            edge_type="has_aspect",
            sentiment=row["sentiment"],
        )
        sentiment_edge_count += 1
    print(f"  added {sentiment_edge_count:,} sentiment edges "
          f"(+ {sentiment_edge_count:,} has_aspect edges)")

    # trust edges: user -> user_trust (self-loop style attribute node, or
    # simplest: store as a node attribute on the user node itself)
    trust_applied = 0
    if len(trust):
        for _, row in trust.iterrows():
            u_node = f"user:{row['user_id']}"
            if u_node in G.nodes:
                G.nodes[u_node]["trust_score"] = row["trust_score"]
                trust_applied += 1
    print(f"  applied trust_score to {trust_applied:,} user nodes "
          f"({'none yet — Day 6 task' if trust_applied == 0 else 'ok'})")

    return G


def main() -> None:
    print("Loading interactions...")
    interactions = load_interactions()

    print("Loading sentiment data...")
    sentiment = load_sentiment()

    print("Loading trust signals...")
    trust = load_trust()

    print("Building graph...")
    G = build_graph(interactions, sentiment, trust)

    GRAPH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_OUTPUT, "wb") as f:
        pickle.dump(G, f)
    print(f"\nSaved graph to {GRAPH_OUTPUT}")

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
    with open(GRAPH_OUTPUT, "rb") as f:
        G_reload = pickle.load(f)
    matches = G_reload.number_of_nodes() == G.number_of_nodes()
    print(f"Reloaded graph: {G_reload.number_of_nodes():,} nodes, "
          f"{G_reload.number_of_edges():,} edges — matches original: {matches}")

if __name__ == "__main__":
    main()