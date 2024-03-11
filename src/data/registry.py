

from src.data.mnist_dataset_opener import train_IterDataPipe_mnist
from src.data.mnist_dataset_opener import train_BulkDataPipe_mnist

DATAPIPE_REGISTRY = {}
DATAPIPE_REGISTRY["MNIST"] = {}
DATAPIPE_REGISTRY["MNIST"]['BulkDataPipe'] = {}
DATAPIPE_REGISTRY["MNIST"]['IterDataPipe'] = {}


DATAPIPE_REGISTRY["MNIST"]['IterDataPipe']["train"] = train_IterDataPipe_mnist
DATAPIPE_REGISTRY["MNIST"]['IterDataPipe']["valid"] = train_IterDataPipe_mnist
DATAPIPE_REGISTRY["MNIST"]['IterDataPipe']["test"] = train_IterDataPipe_mnist


DATAPIPE_REGISTRY["MNIST"]['BulkDataPipe']["train"] = train_BulkDataPipe_mnist
DATAPIPE_REGISTRY["MNIST"]['BulkDataPipe']["valid"] = train_BulkDataPipe_mnist
DATAPIPE_REGISTRY["MNIST"]['BulkDataPipe']["test"] = train_BulkDataPipe_mnist





























