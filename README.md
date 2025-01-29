# Local Retrival MCP

TODO: Description of what this project is.


## Benchmarks

After downloading the datasets with: 
```bash
python scripts/dowload_data.py # Download datasets

# Evaluation on MSMARCO
PYTHONPATH=. python scripts/msmarco_eval.py

# Evaluation on chunking
PYTHONPATH=. python scripts/general_data_eval.py 
```

All benchmarks were conducted using the
[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
embedding model.

### MSMARCO 1M dataset

This dataset is derived from the original
[MSMARCO](https://microsoft.github.io/msmarco/) document retrieval dataset.  To
reduce the search space from the original 8.8 million documents to 1 million,
this dataset includes all documents referenced by MSMARCO test queries,
supplemented with randomly selected documents. To get the same dataset run:

```
python scripts/process_msmarco.py
```

The metrics shown in the following table are calculated as described by the
[ranx documentation](https://amenra.github.io/ranx/metrics/). 

| Method | Precision@5 | nDCG@5 |
|:-|-:|-:|
| BM25 | 0.54 (± 0.29) | 0.11 (± 0.09) |
| Vector Search | 0.67 (± 0.26) | 0.14 (± 0.10) |
| Hybrid Search (bm25 + vector) | **0.76 (± 0.25)** | **0.18 (± 0.16)** |

| Method | Precision@10 | nDCG@10 |
|:-|-:|-:|
| BM25 | 0.57 (± 0.28) | 0.18 (± 0.13) |
| Vector Search | 0.70 (± 0.25) | 0.22 (± 0.13) |
| Hybrid Search (bm25 + vector) | **0.72 (± 0.24)** | **0.25 (± 0.18)** |

| Method | Precision@50 | nDCG@50 |
|:-|-:|-:|
| BM25 | 0.47 (± 0.27) | 0.37 (± 0.18) |
| Vector Search | 0.56 (± 0.29) | 0.46 (± 0.20) |
| Hybrid Search (bm25 + vector) | **0.55 (± 0.25)** | **0.48 (± 0.21)** |


### Chunking Evaluation dataset

This dataset is derived from the technical report "[Evaluating Chunking
Strategies for Retrieval](https://research.trychroma.com/evaluating-chunking)"
by the Chroma Team, published on July 3, 2024. In that research it was used to
evaluate chunking strategies, but since it resembles real life documents quite
well it is a valuable resource for evaluation purposes.

| Method | Recall@3 | Precision@3 | IoU@3 |
|:-|-:|-:|-:|
| BM25 | **0.73 (± 0.43)** | **0.07 (± 0.06)** | **0.07 (± 0.06)** |
| Vector Search | 0.62 (± 0.46) | 0.06 (± 0.06) | 0.06 (± 0.06) |
| Hybrid Search (bm25 + vector) | 0.72 (± 0.43) | 0.06 (± 0.06) | 0.06 (± 0.06) |

| Method | Recall@5 | Precision@5 | IoU@5 |
|:-|-:|-:|-:|
| BM25 | **0.79 (± 0.39)** | **0.05 (± 0.04)** | **0.05 (± 0.04)** |
| Vector Search | 0.70 (± 0.44) | 0.04 (± 0.04) | 0.04 (± 0.04) |
| Hybrid Search (bm25 + vector) | 0.78 (± 0.40) | 0.05 (± 0.04) | 0.04 (± 0.04) |

| Method | Recall@10 | Precision@10 | IoU@10 |
|:-|-:|-:|-:|
| BM25 | **0.83 (± 0.36)** | **0.03 (± 0.02)** | **0.03 (± 0.02)** |
| Vector Search | 0.77 (± 0.41) | 0.02 (± 0.02) | 0.02 (± 0.02) |
| Hybrid Search (bm25 + vector) | 0.83 (± 0.36) | 0.03 (± 0.02) | 0.03 (± 0.02) |
