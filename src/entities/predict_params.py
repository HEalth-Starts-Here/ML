import yaml

from dataclasses import dataclass
from marshmallow_dataclass import class_schema


@dataclass
class PredictPipelineParams:
    model_path: str
    image_path: str
    num_parts: int


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(config_path: str) -> PredictPipelineParams:
    with open(config_path, "r") as input_stream:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
