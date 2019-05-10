from pytorch import training, models

from torch import optim

class ExperimentModel:
    def __init__(self, model, parameters, optimizer):
        self.model = model
        self.parameters = parameters
        self.optimizer = optimizer

def get_model_names():
    return [models.SimpleConv.__name__
        , models.AllConvolutional.__name__
        , models.VGGLike.__name__
        , models.ResNet.__name__]

def get_model(name,dataset,use_cuda):
    def setup_model(model,lr,wd):
        if use_cuda:
            model = model.cuda()
        parameters = training.add_weight_decay(model.named_parameters(), wd)
        optimizer = optim.Adam(parameters, lr=lr)
        return optimizer

    def simple_conv():
        conv_filters = {"mnist": 32, "cifar10": 64, "fashion_mnist": 64}
        fc_filters = {"mnist": 64, "cifar10": 128, "fashion_mnist": 128}
        model = models.SimpleConv(dataset.input_shape, dataset.num_classes,
                                  conv_filters=conv_filters[dataset.name], fc_filters=fc_filters[dataset.name])
        optimizer=setup_model(model,0.001,1e-9)
        rotated_model = models.SimpleConv(dataset.input_shape, dataset.num_classes,
                                          conv_filters=conv_filters[dataset.name],
                                          fc_filters=fc_filters[dataset.name])
        rotated_optimizer = setup_model(rotated_model, 0.001, 1e-9)
        return model, optimizer, rotated_model, rotated_optimizer

    def all_convolutional():
        filters = {"mnist": 16, "mnist_rot": 32, "cifar10": 64, "fashion_mnist": 32}
        model = models.AllConvolutional(dataset.input_shape, dataset.num_classes,
                                        filters=filters[dataset.name])
        optimizer=setup_model(model,1e-3,1e-13)
        rotated_model = models.AllConvolutional(dataset.input_shape, dataset.num_classes,
                                                filters=filters[dataset.name])
        rotated_optimizer = setup_model(rotated_model, 1e-3, 1e-13)

        return model, optimizer, rotated_model, rotated_optimizer

    def vgglike():
        filters = {"mnist": 16, "cifar10": 64,}
        fc= {"mnist": 64,  "cifar10": 128, }
        model = models.VGGLike(dataset.input_shape, dataset.num_classes,
                               conv_filters=filters[dataset.name], fc=fc[dataset.name])
        optimizer=setup_model(model,0.00001,1e-13)
        rotated_model = models.VGGLike(dataset.input_shape, dataset.num_classes,
                                       conv_filters=filters[dataset.name], fc=fc[dataset.name])
        rotated_optimizer = setup_model(rotated_model, 0.00001, 1e-13)

        return model, optimizer, rotated_model, rotated_optimizer
    def resnet():
        resnet_version = {"mnist": models.ResNet18,
                          "cifar10": models.ResNet50,
                          "fashion_mnist": models.ResNet34,
                          }

        model = resnet_version[dataset.name](dataset.input_shape, dataset.num_classes)
        optimizer=setup_model(model,0.00001,1e-13)
        rotated_model = resnet_version[dataset.name](dataset.input_shape, dataset.num_classes)
        rotated_optimizer = setup_model(rotated_model, 0.00001, 1e-13)

        return model, optimizer, rotated_model, rotated_optimizer

    all_models = {models.SimpleConv.__name__: simple_conv,
              models.AllConvolutional.__name__: all_convolutional,
              # pytorch_models.AllConv.__name__: all_conv,
              models.VGGLike.__name__: vgglike,
              models.ResNet.__name__: resnet,
              }
    if name not in all_models :
        raise ValueError(f"Model \"{name}\" does not exist. Choices: {', '.join(all_models .keys())}")
    return all_models [name]()

def get_epochs(dataset,model):
    if model== models.SimpleConv.__name__:
        epochs={'cifar10':70,'mnist':5,'fashion_mnist':12}
        rotated_epochs={'cifar10':120,'mnist':15,'fashion_mnist':60,}
    elif model== models.AllConvolutional.__name__:
        epochs={'cifar10':32,'mnist':15,'fashion_mnist':12}
        rotated_epochs={'cifar10':60,'mnist':50,'fashion_mnist':60,}
    elif model== models.VGGLike.__name__:
        epochs={'cifar10':70,'mnist':15,'fashion_mnist':12,}
        rotated_epochs={'cifar10':150,'mnist':50,'fashion_mnist':60,}
    elif model== models.ResNet.__name__:
        epochs={'cifar10':70,'mnist':15,'fashion_mnist':12}
        rotated_epochs={'cifar10':150,'mnist':50,'fashion_mnist':60}
    else:
        raise ValueError(f"Model \"{model}\" does not exist. Choices: {', '.join(get_model_names().keys())}")
    return epochs[dataset],rotated_epochs[dataset]
