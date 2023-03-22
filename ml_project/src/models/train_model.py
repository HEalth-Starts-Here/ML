
from ultralytics import YOLO

from entities.train_params import (
    TrainingParams,
    TrainingPipelineParams
)

def train_model(
    train_params: TrainingParams,
    pretrained_model_path: str
):
    if train_params.model_type == "yolov8":
        model = YOLO(
            pretrained_model_path
        )  
    # elif train_params.model_type == "GradientBoostingClassifier":
    #     model = GradientBoostingClassifier(
    #         n_estimators=train_params.n_iters
    #     )
    # elif train_params.model_type == "LogisticRegression":
    #     model = LogisticRegression(
    #         random_state=train_params.random_state
    #     )
    else:
        raise NotImplementedError()


    return model