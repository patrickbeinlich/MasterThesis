import os
from importlib import import_module


def get_module(module_name):
    file = import_module(module_name)
    return getattr(file, module_name)


def get_available_models():
    filenames = [f.split('.')[0] for f in os.listdir("../../../../models/") if f.endswith(".py")]

    # Remove the interface
    filenames.remove('IForecastingModel')

    return [x for x in filenames if not x.startswith('_')]


def get_available_multivariant_models():
    models = get_available_models()

    multi_models = []
    for model in models:
        fc_module = get_module(model)
        multi = getattr(fc_module, 'is_multivariate')()
        if multi is True:
            multi_models.append(model)

    return multi_models
