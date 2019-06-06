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

Geoquery

| Method        | Acc<sub>qm</sub>|
| ------------- | ------------- |
| ASN           | 87.1%         |
| SEQ2TREE      | 87.1%         |
| TRANX         | 88.2%         |
| JL16          | 89.2%         |
| **SpatialNLI**| **90.4%**     |

Restaurant

|Method        | Acc<sub>qm</sub>|
|--------------|-----------------|
|**SpatialNLI**|**100%**         |


