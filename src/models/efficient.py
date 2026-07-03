from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict


def get_train_model(num_classes):
    config = get_efficientdet_config("tf_efficientdet_d0")
    config.num_classes = num_classes
    config.image_size = (512, 512)

    net = EfficientDet(config, pretrained_backbone=True)
    return DetBenchTrain(net, config)


def get_predict_model(num_classes):
    config = get_efficientdet_config("tf_efficientdet_d0")
    config.num_classes = num_classes
    config.image_size = (512, 512)

    net = EfficientDet(config, pretrained_backbone=False)
    return DetBenchPredict(net)