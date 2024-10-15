import yaml

from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema

from ml_project.enities.split_params import DatasetParams
from ml_project.enities.feature_params import FeatureParams


@dataclass
class BoostingParams:
    n_estimators: int = field(default=500)
    max_depth: int = field(default=10)
    learning_rate: float = field(default=0.1)


@dataclass
class ModelParams:
    save_path: str
    metric_path: str
    model: str = field(default="GradientBoosting")
    lgbm_params: BoostingParams = field(default_factory=BoostingParams)


@dataclass()
class TrainingParams:
    data: DatasetParams
    features: FeatureParams
    model: ModelParams


@dataclass()
class PredictParams:
    model: str
    data_path: str
    results_path: str


def get_train_params(path: str) -> TrainingParams:
    schema = class_schema(TrainingParams)
    with open(path, "r") as input:
        config_params = yaml.safe_load(input)
        return schema().load(config_params)


def get_predict_params(path: str) -> PredictParams:
    schema = class_schema(PredictParams)
    with open(path, "r") as input:
        config_params = yaml.safe_load(input)
        return schema().load(config_params)
