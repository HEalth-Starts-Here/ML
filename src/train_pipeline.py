import logging
import click
from typing import Tuple

from entities.train_params import (
    read_train_pipeline_params,
    TrainingPipelineParams
)
from src.models.train_yolo import (
    define_model,
    train_model
)
from entities.logger import setup_default_logger




logger = setup_default_logger(
    name="train",
    log_file="train.log",
    level=logging.INFO,
    mode="w"
)

def train_pipeline(config_path: str):
    train_pipeline_params = read_train_pipeline_params(config_path)

    # add handling mlflow
    return run_train_pipeline(train_pipeline_params)

def run_train_pipeline(train_pipeline_params: TrainingPipelineParams) -> Tuple[str, str]:
    logger.info(f"__Start training :: params = {train_pipeline_params}")

    model = define_model(
        train_pipeline_params.train_params,
        train_pipeline_params.pretrained_model_path
    )

    trained_model = train_model(
        model,
        train_pipeline_params.train_params
    )





    # data_frame = read_data(train_pipeline_params.input_data_path)

    # split_data_frame = divide_df_to_sings_marks(
    #     data_frame,
    #     train_pipeline_params.train_dataframe_path
    # )

    # train_df, test_df, train_marks, test_marks = split_train_test_data(
    #     split_data_frame, train_pipeline_params.split_params
    # )

    # logger.info(f"""Dataframe:
    #     train_df  train_marks :: {train_df.shape} {train_marks.shape}
    #     test_df   test_marks  :: {test_df.shape} {test_marks.shape}"""
    # )

    # if not (train_pipeline_params.train_params.scaler is None):
    #     transformer = build_transformer(train_pipeline_params.feature_params)
    #     train_df = transformer.fit_transform(train_df)
    # else:
    #     transformer = None

    # model = train_model(
    #     train_df, train_marks, train_pipeline_params.train_params
    # )

    # inference_pipeline = create_inference_pipeline(model, transformer)

    # y_pred = predict_model(
    #     inference_pipeline,
    #     test_df
    # )

    # metrics = evaluate_model(
    #     y_pred,
    #     test_marks
    # )

    # with open(train_pipeline_params.metric_path, "w") as metric_file:
    #     json.dump(metrics, metric_file)
    # logger.info(f"Metrics :: {metrics}")

    # pp = PrettyPrinter(indent=4, width=40)
    # pp.pprint(metrics)


    # path_to_model = serialize_model(
    #     inference_pipeline, train_pipeline_params.output_model_path
    # )
    # return path_to_model, metrics


@click.command()
@click.argument("config_path")
def main(config_path: str):
    train_pipeline(config_path)

    
if __name__ == "__main__":
    main()