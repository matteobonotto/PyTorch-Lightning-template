

from src.data.mnist_dataset_opener import train_datapipe_mnist

DATAPIPE_REGISTRY = {}

DATAPIPE_REGISTRY["MNIST"] = {}
DATAPIPE_REGISTRY["MNIST"]["train"] = train_datapipe_mnist
DATAPIPE_REGISTRY["MNIST"]["valid"] = train_datapipe_mnist
DATAPIPE_REGISTRY["MNIST"]["test"] = train_datapipe_mnist





























