# SpatialNLI

  SpatialNLI: A Spatial Domain Natural Language Interface to Databases Using Spatial Comprehension

## Usage
To train new model

```python main.py --mode train```

To infer using pre-trained model

```python main.py --mode infer```


## Spatial Comprehension Model
Please refer [Spatial Comprehension](https://github.com/VV123/Spatial-Comprehension).
## Evaluation

We use denotation match accuracy Acc<sub>dm</sub> for evaluation.

Geoquery

| Method        | Acc<sub>dm</sub>|
| ------------- | ------------- |
| ASN           | 87.1%         |
| SEQ2TREE      | 87.1%         |
| TRANX         | 88.2%         |
| JL16          | 89.2%         |
| **SpatialNLI**| [**90.4%**](https://drive.google.com/drive/folders/1GskZI_sPrDbp9yn6YjEtLmmEKtLvT85o)     |

Restaurant

|Method        | Acc<sub>dm</sub>|
|--------------|-----------------|
|**SpatialNLI**|[**100%**](https://drive.google.com/drive/folders/1heNxCCuQ2O8NgfIYFViG0lEk1KwF02Uq)        |


Geoquery + Rest

|Method| Acc<sub>dm</sub>|
|--------------|-----------------|
|**SpatialNLI**|[**90.7**](https://drive.google.com/drive/folders/1ydwkOq-2TokSgL3EmjmJC3i7oYL07PrO)|
