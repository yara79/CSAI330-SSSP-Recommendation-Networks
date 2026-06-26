# Construction and Visualization of Shortest Paths in Recommendation Networks

## Team Members

- Yara Hany 202400477
- Jana Abozaid 202400341
- Nour Afifi 202401540
- Nour Swidan 202400479
- Nadine Yasser 202401895
- Laila Badir 202401135
- Hussein Ahmed 202402007
- Mostafa Thabet 202400632
- Zeyad Haitham 202401698
- Ahmed Abodeeb 202400367

---

## Project Description

This project investigates Single Source Shortest Path (SSSP) algorithms on recommendation networks. Several shortest path algorithms were implemented and evaluated on real-world datasets.

---

## Recommendation Networks

Recommendation networks model interactions between users and items.

Applications include:

- Movie recommendations
- Product recommendations
- Personalized services
- E-commerce systems

---

## Datasets

| Dataset | Nodes | Edges |
|----------|-------|-------|
| MovieLens100K | 943 | 100000 |
| AmazonBooks | 8000 | 50000 |
| Netflix | 17770 | 100480 |
| Yelp | 200000 | 1000000 |
| AmazonFull | 1000000+ | 100000000+ |

---

## Implemented Algorithms

### Dijkstra

Computes shortest paths using a priority queue.

### A*

Uses heuristics to guide the search.

### Bidirectional Dijkstra

Performs simultaneous search from source and destination.

### Contraction Hierarchies

Uses preprocessing for faster shortest path queries.

### Grid Graph Method

Optimized for structured graphs.

### Approximate SSSP

Provides scalable solutions for large graphs.

---

## Experimental Results

### Execution Time Comparison

![Execution Times](figures/execution_times.png)

### Computational Growth

![Growth Curve](figures/growth_curve.png)

---

## Visualization Videos

| Dataset | Algorithm | Video |
|----------|-----------|-------|
| MovieLens | Dijkstra | [Video](videos/movielens_dijkstra.mp4) |
| Netflix | A* | [Video](videos/netflix_astar.mp4) |

---

## How to Run

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook Algorithms_Project_Completed.ipynb
```
