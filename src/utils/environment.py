try:
    import mlflow
    _has_mlflow = True
except ImportError:
    _has_mlflow = False


def requires_mlflow():
    if not _has_mlflow:
        raise ImportError('You need to install mlflow before using this API.')