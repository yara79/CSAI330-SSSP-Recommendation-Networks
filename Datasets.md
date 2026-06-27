# Dataset Details

This page documents the five recommendation network datasets used in this project, their provenance, properties, and how they were processed.

---

## Category: Recommendation Networks

Recommendation networks encode relationships between users and items. They are at the heart of modern personalization systems — Netflix's movie suggestions, Amazon's "customers also bought," and Yelp's business recommendations all rely on graph-based reasoning over these networks.

From an algorithmic perspective, they are interesting because:
- They span **multiple orders of magnitude in size** (hundreds to millions of nodes)
- They are **sparse relative to their node count** (E ≈ O(V) to O(V log V))
- They exhibit **power-law degree distributions** (a few hubs with very high degree)
- They are **naturally connected** in the largest component, making SSSP well-defined

---

## Dataset 1 — rec-MovieLens100K

| Property | Value |
|---|---|
| **Source** | [GroupLens Research](https://grouplens.org/) |
| **Nodes** | 943 |
| **Edges** | ~82,000 (after cleaning) |
| **Scale** | Small |
| **Description** | 100,000 movie ratings by 943 users on 1,682 movies. Modelled as a co-rating graph where two users are connected if they rated the same movie. |

**Why useful:** Small enough to run and visualize all algorithms instantly. Excellent for validating correctness and generating clear animations.

---

## Dataset 2 — rec-AmazonBooks

| Property | Value |
|---|---|
| **Source** | [Amazon Product Data](https://jmcauley.ucsd.edu/data/amazon/) |
| **Nodes** | ~8,000 |
| **Edges** | ~45,000 (LCC) |
| **Scale** | Medium |
| **Description** | Amazon book co-purchase graph. Two books are connected if they were frequently purchased together. Weighted by co-purchase frequency. |

**Why useful:** Medium scale — shows how algorithm runtimes begin to diverge from the MovieLens baseline.

---

## Dataset 3 — rec-Netflix

| Property | Value |
|---|---|
| **Source** | Netflix Prize Dataset |
| **Nodes** | 17,770 |
| **Edges** | 100,480 |
| **Scale** | Medium |
| **Description** | Netflix user-movie rating network. Nodes represent movies; edges connect movies rated by the same users. |

**Why useful:** Real-world famous dataset used in recommendation system research. Scale is large enough to stress-test A\* and CH preprocessing times.

---

## Dataset 4 — rec-Yelp

| Property | Value |
|---|---|
| **Source** | [Yelp Open Dataset](https://www.yelp.com/dataset) |
| **Nodes** | ~200,000 |
| **Edges** | ~900,000 (LCC) |
| **Scale** | Large |
| **Description** | Yelp user-business interaction graph. Users and businesses are nodes; edges represent reviews. |

**Why useful:** Large enough that algorithmic complexity differences become very visible. CH's preprocessing cost dominates here, making A\* the clear winner.

---

## Dataset 5 — rec-AmazonFull

| Property | Value |
|---|---|
| **Source** | Amazon Product Data (full corpus) |
| **Nodes** | 1,000,000+ |
| **Edges** | 100,000,000+ |
| **Scale** | Very Large |
| **Description** | Full Amazon product co-purchase graph across all categories. Represents the scale of real production recommendation systems. |

**Why useful:** Demonstrates the limits of in-memory, single-machine SSSP algorithms. At this scale, distributed or approximate algorithms (e.g., Pruned Landmark Labeling) are required.

---

## Preprocessing Steps Applied to All Datasets

1. **Read** — `pandas.read_csv` with `comment='#'` to skip Network Repository header metadata
2. **Select** — Keep only the first two columns (source node ID, target node ID)
3. **Rename** — `source`, `target`
4. **Deduplicate** — Remove parallel edges
5. **Remove self-loops** — Drop rows where `source == target`
6. **Assign weight** — Uniform `weight = 1` (original interaction data treated as unweighted)
7. **Drop NaN** — Remove any rows with missing values
8. **Build graph** — `nx.from_pandas_edgelist(..., edge_attr='weight')`
9. **Extract LCC** — `max(nx.connected_components(G), key=len)` to ensure full reachability
10. **Persist** — `nx.write_weighted_edgelist()` for algorithm runs

---
