

MODEL_REGISTRY = {
    "conv2d_classif": {
        "class_path": 'src.modules.mnist_modules.conv2d_classif',
        "init_args": {
            "channel_in_list": [1,16,32],
            "channel_out_list": [16, 32, 64],
            "linear_in_features": 150
        },
    },
}



LOSS_REGISTRY = {
    "CrossEntropy": {
        "class_path": 'src.modules.custom_losses.UserCrossEntropyLoss',
        "init_args": {},
    },
}