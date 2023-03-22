

def train_pipeline(config_path: str):
    training_pipline_params = read_training_pipeline_params(config_path)

    # add handling mlflow
    return run_train_pipeline(training_pipline_params)


@click.command()
@click.argument("config_path")
def main(config_path: str):
    train_pipeline(config_path)

    
if __name__ == "__main__":
    main()