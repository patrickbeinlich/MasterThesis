from enum import Enum


class FileTypes(Enum):
    MODEL = "model"
    DATA = "data"


class ModelActionsEnum(Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"


class InfoActionsEnum(Enum):
    MODELS = "models"
    MULTI_MODELS = "multivariante_models"
