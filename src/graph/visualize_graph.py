"""Visualize specifically the sentiment/aspect edges to confirm they're real."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = REPO_ROOT / "data" / "processed" / "video_games" / "graph.gpickle"
OUTPUT_PATH = REPO_ROOT / "results" / "graph_sentiment_edges.png"


def main() -> None:
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    aspect_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "aspect"]
    nodes_to_keep = set(aspect_nodes)
    for a in aspect_nodes:
        nodes_to_keep.update(G.predecessors(a))  # items pointing to this aspect
    for n in list(nodes_to_keep):
        if G.nodes[n].get("node_type") == "item":
            nodes_to_keep.update(G.predecessors(n))  # users pointing to this item

    subG = G.subgraph(nodes_to_keep)
    print(f"Sentiment-focused sub-graph: {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges")

    color_map = {"user": "skyblue", "item": "lightgreen", "aspect": "salmon"}
    colors = [color_map[subG.nodes[n]["node_type"]] for n in subG.nodes]

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(subG, seed=1, k=0.8)
    nx.draw(subG, pos, node_color=colors, node_size=400, with_labels=True,
            font_size=6, edge_color="gray", arrows=True, arrowsize=10)
    plt.title("The 4 sentiment/aspect edges and their connected users/items")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()