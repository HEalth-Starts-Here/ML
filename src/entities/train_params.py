from dataclasses import dataclass
from typing import Optional

from marshmallow_dataclass import class_schema
import yaml


@dataclass
class TrainingParams:
    model_type: Optional[str] = "yolov8"  # train model
    epochs: Optional[int] = 10  # number of epochs
    imgsz: Optional[int] = 640
    optimizer: Optional[str] = "SGD"
    batch: Optional[int] = 16
    seed: Optional[int] = 5  # for reproducing research
    project: Optional[str] = "data/result"  # save dir
    name: Optional[str] = "train"  # save dir
    device: Optional[str] = None


@dataclass()
class TrainingPipelineParams:
    input_data_path: str  # path to splited dataset
    pretrained_model_path: str  # pretrained model download from net, need save path
    output_model_path: str  # save fine tuned model
    output_metric_path: str  # save result metric
    output_data_path: str
    split_params: SplitParams
    train_params: TrainingParams
    train_dataframe_path: Optional[str] = "data/raw/predict_dataset.csv"
    scaler: Optional[str] = None


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_train_pipeline_params(config_path: str) -> TrainingPipelineParams:
    with open(config_path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
