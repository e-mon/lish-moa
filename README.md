# Mechanisms of Action (MoA) Prediction

4th place solution for Mechanisms of Action (MoA) Prediction https://www.kaggle.com/c/lish-moa/discussion/200616

Solution summary: [here](https://www.kaggle.com/c/lish-moa/discussion/200808)

Kernel: [here](https://www.kaggle.com/kento1993/nn-svm-tabnet-xgb-with-pca-cnn-stacking-without-pp)

## B2 Requirements

- HW
    - These codes are runnable at kaggle notebook instance spec
        - CPU: 2 core
        - Memory: 13GB
        - GPU: 16GB (V100)
        - Disk: 20GB
- OS/platform
    - it's feasible under the following docker image environment
        - `gcr.io/kaggle-gpu-images/python@sha256:c87ecab24a46ae164699eab2d5627a2a09fea4462dabf57c3034cd43c46c7cb8`
- Train & Inference Model
    - Train & Inference processes are not separeted as a file. please check our notebook.
- Side effects
    - Nothing
- Key assumptions
    - It is assumed that all of our codes are executed on Kaggle notebook or Docker environment above mentioned. please check `docker-compose.yaml` file to reproduct Kaggle notebook environment in your pc via Docker image.

## B3 Configuration files
- Nothing


## B4 Requirements.txt
- All of libraries needed are included in Kaggle Docker image.

## B5 directory_structure.txt & B6 SETTINGS.json

directory structure
```
.
├── input
│   └── lish-moa # input data
└── working
    ├── src      # libraries (https://github.com/e-mon/lish-moa)
    └── cache    # pretrained models (https://www.kaggle.com/eemonn/moa-cache)
```

input & output directory is specified in `docker-compose.yaml`.
If you want to change direcotry, please overwrite following paths.
```
    volumes:
      - ./working:/working
      - ./input:/input
```

## Setup

Since these codes are designed to be executed on Kaggle Kernel, so first get the BASE64-encoded codes by running the following command and paste it your notebook.

(refer to: https://github.com/lopuhin/kaggle-imet-2019)
```shell
$ make build
```