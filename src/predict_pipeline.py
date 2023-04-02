import click
import torch
import torchvision
import numpy as np

from PIL import Image

from src.entities.predict_params import read_predict_pipeline_params
from src.models.iqa_models import HyperNet, TargetNet


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def predict_pipeline(params_path: str) -> float:
    params = read_predict_pipeline_params(params_path)

    model_hyper = HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
    model_hyper.train(False)
    model_hyper.load_state_dict((torch.load(params.model_path)))

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((512, 384)),
            torchvision.transforms.RandomCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )

    # random crop 10 patches and calculate mean quality score
    # quality score ranges from 0-100, a higher score indicates a better quality

    pred_scores = []
    for _ in range(params.num_parts):
        img = pil_loader(params.image_path)
        img = transforms(img)
        img = torch.tensor(img.cuda()).unsqueeze(0)
        weights = model_hyper(img)

        model_target = TargetNet(weights).cuda()
        for param in model_target.parameters():
            param.requires_grad = False

        pred = model_target(weights["target_in_vec"])
        pred_scores.append(float(pred.item()))
    score = np.mean(pred_scores)
    return score


@click.command()
@click.argument("config_path")
def main(config_path: str):
    score = predict_pipeline(config_path)
    print(score)


if __name__ == "__main__":
    main()
