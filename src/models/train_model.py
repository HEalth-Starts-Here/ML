
from ultralytics import YOLO

from entities.train_params import (
    TrainingParams,
    TrainingPipelineParams
)

def define_model(
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

def train_model(
    model,
    train_params: TrainingParams        
):
    if train_params.model_type == "yolov8":
        model.train(

        )


        model.train(
            data=f'{DATASET_PATH}/data.yaml'  # data path
            ,epochs=1
            ,imgsz=512
            ,optimizer="AdamW"
            ,batch=16 
            ,seed=567 
            ,project=f"{TRAIN_RESULT_PATH}"  # save directory path  
            ,name="train"  # spam default "train" "train2" "train3" ...
            ,device=None
        )