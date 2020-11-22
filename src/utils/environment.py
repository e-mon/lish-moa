try:
    import mlflow
    _has_mlflow = True
except ImportError:
    _has_mlflow = False


def requires_mlflow():
    if not _has_mlflow:
        raise ImportError('You need to install mlflow before using this API.')


try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False


def get_device():
    if _has_torch:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        'cpu'
