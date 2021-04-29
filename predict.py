import os
import json
import pandas as pd
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
import torchvision as tv
import matplotlib as mpl

mpl.use("cairo")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.autograd import Variable

from argparse import ArgumentParser
from data_processing import build_prediction_csv
from dataset import PneumoniaDataset
from model import PneumoniaUNET
from metrics import prediction_string, parse_boxes


def make_parser():
    parser = ArgumentParser(
        description="Run inference for Pneumonia detection"
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=f"{os.environ.get('TRAINML_DATA_PATH')}",
        help="path to image files",
    )
    parser.add_argument(
        "--batch-size",
        "--bs",
        type=int,
        default=10,
        help="number of images for each iteration",
    )
    parser.add_argument(
        "--rescale-factor",
        "--rf",
        type=int,
        default=4,
        help="resize factor to reduce image size",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=f"{os.environ.get('TRAINML_MODEL_PATH')}/final.pth.tar",
        help="path to model file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"{os.environ.get('TRAINML_OUTPUT_PATH')}",
        help="path to save output images and data",
    )

    return parser


def rescale_box_coordinates(box, rescale_factor):
    x, y, w, h = box
    x = int(round(x / rescale_factor))
    y = int(round(y / rescale_factor))
    w = int(round(w / rescale_factor))
    h = int(round(h / rescale_factor))
    return [x, y, w, h]


def draw_boxes(predicted_boxes, confidences, target_boxes, ax, angle=0):
    if len(predicted_boxes) > 0:
        for box, c in zip(predicted_boxes, confidences):
            # extracting individual coordinates
            x, y, w, h = box
            # create a rectangle patch
            patch = Rectangle(
                (x, y),
                w,
                h,
                color="red",
                ls="dashed",
                angle=angle,
                fill=False,
                lw=4,
                joinstyle="round",
                alpha=0.6,
            )
            # get current axis and draw rectangle
            ax.add_patch(patch)
            # add confidence value in annotation text
            ax.text(
                x + w / 2.0,
                y - 5,
                "{:.2}".format(c),
                color="red",
                size=20,
                va="center",
                ha="center",
            )
    if len(target_boxes) > 0:
        for box in target_boxes:
            # rescale and extract box coordinates
            x, y, w, h = box
            # create a rectangle patch
            patch = Rectangle(
                (x, y),
                w,
                h,
                color="red",
                angle=angle,
                fill=False,
                lw=4,
                joinstyle="round",
                alpha=0.6,
            )
            # get current axis and draw rectangle
            ax.add_patch(patch)

    return ax


def save_image_prediction(file, img, prediction, predicted_boxes, confidences):
    plt.imshow(
        img[0], cmap=mpl.cm.gist_gray
    )  # [0] is the channel index (here there's just one channel)
    plt.imshow(prediction[0], cmap=mpl.cm.jet, alpha=0.5)
    draw_boxes(predicted_boxes, confidences, [], plt.gca())
    plt.savefig(file)
    plt.close()


def predict(model, dataloader):

    # set model to evaluation mode
    model.eval()

    predictions = {}

    for i, (pred_batch, pIds) in enumerate(dataloader):
        print("Predicting batch {} / {}.".format(i + 1, len(dataloader)))
        # Convert torch tensor to Variable
        pred_batch = Variable(pred_batch).cuda()

        # compute output
        output_batch = model(pred_batch)
        sig = nn.Sigmoid().cuda()
        output_batch = sig(output_batch)
        output_batch = output_batch.data.cpu().numpy()
        for pId, output in zip(pIds, output_batch):
            predictions[pId] = output

    return predictions


def get_prediction_string_for_prediction(
    prediction, threshold, rescale_factor
):
    predicted_boxes, confidences = parse_boxes(
        prediction, threshold=threshold, connectivity=None
    )
    predicted_boxes = [
        rescale_box_coordinates(box, 1 / rescale_factor)
        for box in predicted_boxes
    ]
    return prediction_string(predicted_boxes, confidences)


def run_inference(args):
    build_prediction_csv(args.data, args.output)
    df = pd.read_csv(args.output + "/predict.csv")
    pIds = df["patientId"].unique()
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])
    dataset = PneumoniaDataset(
        root=args.data,
        pIds=pIds,
        predict=True,
        boxes=None,
        rescale_factor=args.rescale_factor,
        transform=transform,
        rotation_angle=0,
        warping=False,
    )
    loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=False
    )
    model = PneumoniaUNET().cuda()
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint["state_dict"])
    best_threshold = checkpoint.get("best_threshold") or 0.2
    predictions = predict(model, loader)
    print("Predicted {} images.".format(len(predictions)))

    os.makedirs(f"{args.output}/images", exist_ok=True)
    annotations = dict()
    for i in range(len(dataset)):
        img, pId = dataset[i]
        prediction = predictions[pId]
        predicted_boxes, confidences = parse_boxes(
            prediction, threshold=best_threshold, connectivity=None
        )
        save_image_prediction(
            f"{args.output}/images/{pId}.png",
            img,
            prediction,
            predicted_boxes,
            confidences,
        )
        annotations[pId] = get_prediction_string_for_prediction(
            prediction, best_threshold, args.rescale_factor
        )

    with open(f"{args.output}/annotations.json", "w") as f:
        f.write(json.dumps(annotations))


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    print(args)
    run_inference(args)
