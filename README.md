### This is a project to make local searching in documents efficient and good

Inital benchmark results on the MSMARCO dataset:

| Model         |   Hits   | Recall |   P   |  P@5  |  P@10  | Storage | Speed         |
|---------------|---------:|-------:|------:|------:|-------:|--------:|---------------|
| BM25          |  37.698  |  0.522 | 0.377 | 0.619 |  0.584 |   ?     | 40.49 query/s |
| vector-search |  43.535  |  0.599 | 0.435 | 0.819 |  0.784 |  2.8 GB |  2.93 query/s |

