## Getting started
start Kaggle docker image

```shell
$ cd /path/to/lish-moa && docker-compose up -d
```

## Reproduce model prediction
Since the Jupyter service is already running, you can either run the notebook directly, or run following CLI commands:
```shell
# Attach jupyter container
$ docker-compose exec jupyter /bin/bash
# move notebook directory & run
$ cd working && papermill nn-svm-tabnet-xgb-with-pca-cnn-stacking-without-pp.ipynb output.ipynb
```
