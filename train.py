import time
import numpy as np
import pandas as pd
import torch
import os
from argparse import ArgumentParser
from torch.autograd import Variable
import torchvision as tv
from torch.utils.data import DataLoader


from metrics import RunningAverage, average_precision_batch
from model import save_checkpoint, PneumoniaUNET, BCEWithLogitsLoss2d
from dataset import PneumoniaDataset, get_boxes_per_patient


def make_parser():
    parser = ArgumentParser(
        description="Train UNET segmentation model for Pneumonia detection"
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=f"{os.environ.get('TRAINML_DATA_PATH')}/stage_2_train_images",
        help="path to image files",
    )
    parser.add_argument(
        "--labels",
        "-l",
        type=str,
        default=f"{os.environ.get('TRAINML_MODEL_PATH')}/train.csv",
        help="path to labels file for training",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=20,
        help="number of epochs for training",
    )
    parser.add_argument(
        "--batch-size",
        "--bs",
        type=int,
        default=6,
        help="number of examples for each iteration",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="enable debug mode",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="manually set random seed for torch",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="path to model checkpoint file",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=os.environ.get("TRAINML_OUTPUT_PATH"),
        help="save model checkpoints in the specified directory",
    )

    # Hyperparameters
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=0.5,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--learning-rate-decay",
        "--lrd",
        type=float,
        default=0.5,
        help="learning rate decay for Adagrad optimizer",
    )
    parser.add_argument(
        "--optimizer",
        "-o",
        type=str,
        default="adam",
        choices=["adam", "adamw", "adamax", "sgd", "adagrad"],
    )
    parser.add_argument(
        "--momentum",
        "-m",
        type=float,
        default=0.9,
        help="momentum argument for Batch Norm Layer and SGD optimizer",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-05,
        help="eps argument for Batch Norm Layer and Adam optimizer",
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        type=float,
        default=0.0005,
        help="momentum argument for all optimizers",
    )
    parser.add_argument(
        "--rescale-factor",
        "--rf",
        type=int,
        default=4,
        help="resize factor to reduce image size",
    )
    parser.add_argument(
        "--alpha-leaky",
        "--al",
        type=int,
        default=0.03,
        help="alpha for LeakyReLU",
    )

    return parser


def train_one(
    model,
    dataloader,
    optimizer,
    loss_fn,
    num_steps,
    pId_boxes_dict,
    rescale_factor,
    shape,
    save_summary_steps=5,
):
    # set model to train model
    model.train()

    loss_avg = RunningAverage()

    loss_avg_t_hist_ep, loss_t_hist_ep, prec_t_hist_ep = [], [], []

    # iterate over batches
    start = time.time()

    for i, (input_batch, labels_batch, pIds_batch) in enumerate(dataloader):
        # break loop after num_steps batches (useful for debugging)
        if i > num_steps:
            break
        # Convert torch tensor to Variable
        input_batch = Variable(input_batch).cuda()
        labels_batch = Variable(labels_batch).cuda()

        # compute output
        optimizer.zero_grad()
        output_batch = model(input_batch)

        # compute loss
        loss = loss_fn(output_batch, labels_batch)

        # compute gradient and do optimizer step
        loss.backward()
        optimizer.step()

        # update loss running average
        loss_avg.update(loss.item())
        loss_t_hist_ep.append(loss.item())
        loss_avg_t_hist_ep.append(loss_avg())

        # Evaluate summaries only once in a while
        if i % save_summary_steps == 0:
            # extract data from torch Variable, move to cpu
            output_batch = output_batch.data.cpu().numpy()
            # compute average precision on this batch
            prec_batch = average_precision_batch(
                output_batch, pIds_batch, pId_boxes_dict, rescale_factor, shape
            )
            prec_t_hist_ep.append(prec_batch)
            # log results
            summary_batch_string = "batch loss = {:05.7f} ;  ".format(
                loss.item()
            )
            summary_batch_string += "average loss = {:05.7f} ;  ".format(
                loss_avg()
            )
            summary_batch_string += "batch precision = {:05.7f} ;  ".format(
                prec_batch
            )
            print(
                "--- Train batch {} / {}: ".format(i, num_steps)
                + summary_batch_string
            )
            delta_time = time.time() - start
            print(
                "    {} batches processed in {:.2f} seconds".format(
                    save_summary_steps, delta_time
                )
            )
            start = time.time()

    # log epoch summary
    metrics_string = "average loss = {:05.7f} ;  ".format(loss_avg())
    print("- Train epoch metrics summary: " + metrics_string)

    return loss_avg_t_hist_ep, loss_t_hist_ep, prec_t_hist_ep


def evaluate(
    model,
    dataloader,
    loss_fn,
    num_steps,
    pId_boxes_dict,
    rescale_factor,
    shape,
):

    # set model to evaluation mode
    model.eval()

    losses = []
    precisions = []

    # compute metrics over the dataset
    start = time.time()
    for i, (input_batch, labels_batch, pIds_batch) in enumerate(dataloader):
        # break loop after num_steps batches (useful for debugging)
        if i > num_steps:
            break
        # Convert torch tensor to Variable
        input_batch = Variable(input_batch).cuda()
        labels_batch = Variable(labels_batch).cuda()

        # compute model output
        output_batch = model(input_batch)
        # compute loss of batch
        loss = loss_fn(output_batch, labels_batch)
        losses.append(loss.item())

        # extract data from torch Variable, move to cpu
        output_batch = output_batch.data.cpu()
        # compute individual precisions of batch images
        prec_batch = average_precision_batch(
            output_batch,
            pIds_batch,
            pId_boxes_dict,
            rescale_factor,
            shape,
            return_array=True,
        )
        for p in prec_batch:
            precisions.append(p)
        print("--- Validation batch {} / {}: ".format(i, num_steps))

    # compute mean of all metrics in summary
    metrics_mean = {
        "loss": np.nanmean(losses),
        "precision": np.nanmean(np.asarray(precisions)),
    }
    metrics_string = "average loss = {:05.7f} ;  ".format(metrics_mean["loss"])
    metrics_string += "average precision = {:05.7f} ;  ".format(
        metrics_mean["precision"]
    )
    print("- Eval metrics : " + metrics_string)
    delta_time = time.time() - start
    print("  Evaluation run in {:.2f} seconds.".format(delta_time))

    return metrics_mean


def train_and_evaluate(
    model,
    train_dataloader,
    val_dataloader,
    lr_init,
    optimizer_type,
    lr_decay,
    momentum,
    eps,
    wd,
    loss_fn,
    num_epochs,
    num_steps_train,
    num_steps_eval,
    pId_boxes_dict,
    rescale_factor,
    shape,
    save_file=None,
    restore_file=None,
):

    # reload weights from restore_file if specified
    if restore_file is not None:
        checkpoint = torch.load(restore_file)
        model.load_state_dict(checkpoint["state_dict"])

    best_val_loss = 1e15
    best_val_prec = 0.0
    best_loss_model = None
    best_prec_model = None

    loss_t_history = []
    loss_v_history = []
    loss_avg_t_history = []
    prec_t_history = []
    prec_v_history = []

    for epoch in range(num_epochs):
        start = time.time()

        # define the optimizer
        lr = lr_init * 0.5 ** float(
            epoch
        )  # reduce the learning rate at each epoch
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps)
        # Run one epoch
        print(
            "Epoch {}/{}. Learning rate = {:05.3f}.".format(
                epoch + 1, num_epochs, lr
            )
        )

        # train model for a whole epoc (one full pass over the training set)
        loss_avg_t_hist_ep, loss_t_hist_ep, prec_t_hist_ep = train_one(
            model,
            train_dataloader,
            optimizer,
            loss_fn,
            num_steps_train,
            pId_boxes_dict,
            rescale_factor,
            shape,
        )
        loss_avg_t_history += loss_avg_t_hist_ep
        loss_t_history += loss_t_hist_ep
        prec_t_history += prec_t_hist_ep

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(
            model,
            val_dataloader,
            loss_fn,
            num_steps_eval,
            pId_boxes_dict,
            rescale_factor,
            shape,
        )

        val_loss = val_metrics["loss"]
        val_prec = val_metrics["precision"]

        loss_v_history += len(loss_t_hist_ep) * [val_loss]
        prec_v_history += len(prec_t_hist_ep) * [val_prec]

        is_best_loss = val_loss <= best_val_loss
        is_best_prec = val_prec >= best_val_prec

        if is_best_loss:
            print("- Found new best loss: {:.4f}".format(val_loss))
            best_val_loss = val_loss
            best_loss_model = model
        if is_best_prec:
            print("- Found new best precision: {:.4f}".format(val_prec))
            best_val_prec = val_prec
            best_prec_model = model

        # Save best weights based on best_val_loss and best_val_prec
        if save_checkpoint:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optim_dict": optimizer.state_dict(),
                },
                save_file,
                is_best=is_best_loss,
                metric="loss",
            )
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optim_dict": optimizer.state_dict(),
                },
                save_file,
                is_best=is_best_prec,
                metric="prec",
            )

        delta_time = time.time() - start
        print("Epoch run in {:.2f} minutes".format(delta_time / 60.0))

    histories = {
        "loss avg train": loss_avg_t_history,
        "loss train": loss_t_history,
        "precision train": prec_t_history,
        "loss validation": loss_v_history,
        "precision validation": prec_v_history,
    }
    best_models = {
        "best loss model": best_loss_model,
        "best precision model": best_prec_model,
    }

    return histories, best_models


def train(args):
    min_box_area = 10000
    original_image_shape = 1024
    validation_frac = 0.10

    df_train = pd.read_csv(args.labels)
    df_train = df_train.sample(
        frac=1, random_state=args.seed
    )  # .sample(frac=1) does the shuffling
    pIds = [pId for pId in df_train["patientId"].unique()]

    pIds_valid = pIds[: int(round(validation_frac * len(pIds)))]
    pIds_train = pIds[int(round(validation_frac * len(pIds))) :]
    print(
        "{} patient IDs shuffled and {}% of them used in validation set.".format(
            len(pIds), validation_frac * 100
        )
    )
    print(
        "{} images went into train set and {} images went into validation set.".format(
            len(pIds_train), len(pIds_valid)
        )
    )

    pId_boxes_dict = {}
    for pId in (
        df_train.loc[(df_train["Target"] == 1)]["patientId"].unique().tolist()
    ):
        pId_boxes_dict[pId] = get_boxes_per_patient(df_train, pId)
    print(
        "{} ({:.1f}%) images have target boxes.".format(
            len(pId_boxes_dict), 100 * (len(pId_boxes_dict) / len(pIds))
        )
    )

    transform = tv.transforms.Compose([tv.transforms.ToTensor()])

    # create datasets
    dataset_train = PneumoniaDataset(
        root=args.data,
        pIds=pIds_train,
        predict=False,
        boxes=pId_boxes_dict,
        rescale_factor=args.rescale_factor,
        transform=transform,
        rotation_angle=3,
        warping=True,
        seed=args.seed,
    )

    dataset_valid = PneumoniaDataset(
        root=args.data,
        pIds=pIds_valid,
        predict=False,
        boxes=pId_boxes_dict,
        rescale_factor=args.rescale_factor,
        transform=transform,
        rotation_angle=0,
        warping=False,
        seed=args.seed,
    )

    # define the dataloaders with the previous dataset
    loader_train = DataLoader(
        dataset=dataset_train, batch_size=args.batch_size, shuffle=True
    )

    loader_valid = DataLoader(
        dataset=dataset_valid, batch_size=args.batch_size, shuffle=True
    )

    # define an instance of the model
    model = PneumoniaUNET(
        bn_momentum=args.momentum,
        eps=args.eps,
        alpha_leaky=args.alpha_leaky,
    ).cuda()
    # define the loss function
    loss_fn = BCEWithLogitsLoss2d().cuda()

    num_epochs = 2 if args.debug else args.epochs
    num_steps_train = 50 if args.debug else len(loader_train)
    num_steps_eval = 10 if args.debug else len(loader_valid)

    shape = int(round(original_image_shape / args.rescale_factor))

    histories, best_models = train_and_evaluate(
        model,
        loader_train,
        loader_valid,
        args.lr,
        args.optimizer,
        args.learning_rate_decay,
        args.momentum,
        args.eps,
        args.weight_decay,
        loss_fn,
        num_epochs,
        num_steps_train,
        num_steps_eval,
        pId_boxes_dict,
        args.rescale_factor,
        shape,
        save_file=args.save,
        restore_file=args.checkpoint,
    )
    print(histories)
    print(best_models)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    print(args)
    train(args)