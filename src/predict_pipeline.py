import click
import spacy
import pytextrank

from src.entities.logger import setup_default_logger
from src.entities.predict_pipeline_params import (
    read_predicting_pipeline_params,
    PredictingPipelineParams
)
from src.data.make_dataset import (
    read_data,
    save_data
)


logger = setup_default_logger("main", "predict.log")


def predict_pipeline(config_path: str):
    predicting_pipline_params = read_predicting_pipeline_params(config_path)

    return run_predict_pipeline(predicting_pipline_params)


def run_predict_pipeline(predicting_pipeline_params: PredictingPipelineParams):
    logger.info(f"__Start predicting :: params = {predicting_pipeline_params}")

    raw_txt = read_data(predicting_pipeline_params.text_path)

    # with open(predicting_pipeline_params.text_path, "r") as fd:
    #     raw_txt = fd.read()

    nlp = spacy.load(predicting_pipeline_params.nlp_sum_type)
    nlp.add_pipe("textrank", last=True)
    doc = nlp(raw_txt)

    save_data(predicting_pipeline_params.output_result_path, doc, predicting_pipeline_params.limit_sentence)

    # with open(predicting_pipeline_params.output_result_path, "w") as fd:
    #     for sent in doc._.textrank.summary(limit_phrases=15, limit_sentences=2):
    #         fd.write(str(sent))


@click.command()
@click.argument("config_path")
def main(config_path: str):
    predict_pipeline(config_path)

if __name__ == "__main__":
    main()
