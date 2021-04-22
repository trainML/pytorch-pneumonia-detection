from torch import nn
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from torch.autograd import Variable


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