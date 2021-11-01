# SpatialNLI

  [SpatialNLI: A Spatial Domain Natural Language Interface to Databases Using Spatial Comprehension](https://arxiv.org/pdf/1908.10917.pdf)
  This repo contains annotation and augmentation methods  
      
## Dependencies
  - TF 1.4
  - python 2.7

## Usage

To annotate Geo880

```python utils/annotate/annotate_geo.py```

To annotate Rest

```python utils/annotate/annotate_rest.py```

To Build data

```python utils/data_manager.py --data 'geo'```

```python utils/data_manager.py --data 'rest'```


To augment Geo880

```python utils/augmentation/augmentation_geo.py```

To augment Rest

```python utils/augmentation/augmentation_rest.py```
