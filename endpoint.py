import os
import json
import torch
import tempfile
import base64
from torch import nn
import torchvision as tv
from torch.autograd import Variable

from dataset import load_and_prepare_image
from model import PneumoniaUNET
from predict import (
    save_image_prediction,
    get_prediction_string_for_prediction,
    parse_boxes,
)


## Load model in global scope so it's preloaded for API requests
transform = tv.transforms.Compose([tv.transforms.ToTensor()])
model = PneumoniaUNET().cuda()
checkpoint = torch.load(
    os.environ.get("MODEL_CHECKPOINT")
    or f"{os.environ.get('TRAINML_MODEL_PATH')}/final.pth.tar"
)
model.load_state_dict(checkpoint["state_dict"])
best_threshold = checkpoint.get("best_threshold") or 0.2
model.eval()

rescale_factor = os.environ.get("IMAGE_RESCALE_FACTOR") or 4


def predict(img):
    # set model to evaluation mode

    pred = Variable(img).cuda(non_blocking=True)
    # compute output
    output = model(pred)
    sig = nn.Sigmoid().cuda()
    output = sig(output)
    output = output.data.cpu().numpy()
    return output


def predict_file(input_file, output_file):
    img, _, _ = load_and_prepare_image(
        input_file, rescale_factor=rescale_factor, transform=transform
    )
    input = img.unsqueeze(0)
    prediction = predict(input)
    predicted_boxes, confidences = parse_boxes(
        prediction[0], threshold=best_threshold, connectivity=None
    )
    save_image_prediction(
        output_file,
        img,
        prediction[0],
        predicted_boxes,
        confidences,
    )
    annotations = get_prediction_string_for_prediction(
        prediction[0], best_threshold, rescale_factor
    )
    return annotations


def get_prediction(pId, dicom):
    fd, input_file = tempfile.mkstemp()
    fd2, output_file = tempfile.mkstemp()
    os.close(fd2)
    with open(fd, "wb") as f:
        f.write(base64.b64decode(dicom))

    annotations = predict_file(input_file, output_file)
    print(output_file)
    with open(output_file, "rb") as f:
        image = base64.b64encode(f.read()).decode("utf-8")

    print(image)
    os.remove(input_file)
    # os.remove(output_file)

    return {pId: dict(annotations=annotations, image=image)}
