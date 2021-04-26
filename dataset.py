import pydicom
import numpy as np
import os
import torchvision as tv
import PIL
import torch
from torch.utils.data.dataset import Dataset as torchDataset
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
from skimage.exposure import rescale_intensity
import warnings


def get_boxes_per_patient(df, pId):
    """
    Given the dataset and one patient ID,
    return an array of all the bounding boxes and their labels associated with that patient ID.
    Example of return:
    array([[x1, y1, width1, height1],
           [x2, y2, width2, height2]])
    """

    boxes = (
        df.loc[df["patientId"] == pId][["x", "y", "width", "height"]]
        .astype("int")
        .values.tolist()
    )
    return boxes


# define a MinMaxScaler function for the images
def imgMinMaxScaler(img, scale_range):
    """
    :param img: image to be rescaled
    :param scale_range: (tuple) (min, max) of the desired rescaling
    """
    warnings.filterwarnings("ignore")
    img = img.astype("float64")
    img_std = (img - np.min(img)) / (np.max(img) - np.min(img))
    img_scaled = img_std * float(scale_range[1] - scale_range[0]) + float(
        scale_range[0]
    )
    # round at closest integer and transform to integer
    img_scaled = np.rint(img_scaled).astype("uint8")

    return img_scaled


# define a "warping" image/mask function
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       Code adapted from https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    """
    assert len(image.shape) == 2, "Image must have 2 dimensions."

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = (
        gaussian_filter(
            (random_state.rand(*shape) * 2 - 1),
            sigma,
            mode="constant",
            cval=0,
        )
        * alpha
    )
    dy = (
        gaussian_filter(
            (random_state.rand(*shape) * 2 - 1),
            sigma,
            mode="constant",
            cval=0,
        )
        * alpha
    )

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    image_warped = map_coordinates(image, indices, order=1).reshape(shape)

    return image_warped


# define the data generator class
class PneumoniaDataset(torchDataset):
    """
    Pneumonia dataset that contains radiograph lung images as .dcm.
    Each patient has one image named patientId.dcm.
    """

    def __init__(
        self,
        root,
        pIds,
        predict,
        boxes,
        rescale_factor=1,
        transform=None,
        rotation_angle=0,
        warping=False,
        seed=42,
    ):
        """
        :param root: it has to be a path to the folder that contains the dataset folders
        :param subset: 'train' or 'test'
        :param pIds: list of patient IDs
        :param predict: boolean, if true returns images and target labels, otherwise returns only images
        :param boxes: a {patientId : list of boxes} dictionary (ex: {'pId': [[x1, y1, w1, h1], [x2, y2, w2, h2]]}
        :param rescale_factor: image rescale factor in network (image shape is supposed to be square)
        :param transform: transformation applied to the images and their target masks
        :param rotation_angle: float, defines range of random rotation angles for augmentation (-rotation_angle, +rotation_angle)
        :param warping: boolean, whether applying augmentation warping to image
        """

        # initialize variables
        self.root = os.path.expanduser(root)
        self.pIds = pIds
        self.predict = predict
        self.boxes = boxes
        self.rescale_factor = rescale_factor
        self.transform = transform
        self.rotation_angle = rotation_angle
        self.warping = warping
        self.random_state = np.random.RandomState(seed=seed)

        self.data_path = f"{self.root}/"

    def __getitem__(self, index):
        # get the corresponding pId
        pId = self.pIds[index]
        # load dicom file as numpy array
        img = pydicom.dcmread(
            os.path.join(self.data_path, pId + ".dcm")
        ).pixel_array
        # summary = dict(
        #     row_range=np.ptp(np.ptp(img, axis=0)),
        #     column_range=np.ptp(np.ptp(img, axis=1)),
        #     mean=np.mean(img),
        #     shape=img.shape,
        # )
        # print("original image", summary)
        # check if image is square
        if img.shape[0] != img.shape[1]:
            raise RuntimeError(
                "Image shape {} should be square.".format(img.shape)
            )
        original_image_shape = img.shape[0]
        # calculate network image shape
        image_shape = original_image_shape / self.rescale_factor
        # check if image_shape is an integer
        if image_shape != int(image_shape):
            raise RuntimeError(
                f"Network image shape should be an integer. Was {image_shape}"
            )
        image_shape = int(image_shape)
        # resize image
        # IMPORTANT: skimage resize function rescales the output from 0 to 1, and pytorch doesn't like this!
        # One solution would be using torchvision rescale function (but need to differentiate img and target transforms)
        # Here I use skimage resize and then rescale the output again from 0 to 255
        img = resize(img, (image_shape, image_shape), mode="reflect")
        # recale image from 0 to 255
        img = imgMinMaxScaler(img, (0, 255))
        # image warping augmentation
        if self.warping:
            img = elastic_transform(
                img,
                image_shape * 2.0,
                image_shape * 0.1,
                random_state=self.random_state,
            )
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        # apply rotation augmentation
        if self.rotation_angle > 0:
            angle = self.rotation_angle * (
                2 * np.random.random_sample() - 1
            )  # generate random angle
            img = tv.transforms.functional.to_pil_image(img)
            img = tv.transforms.functional.rotate(
                img, angle, resample=PIL.Image.BILINEAR
            )

        # apply transforms to image
        if self.transform is not None:
            img = self.transform(img)

        # summary = dict(
        #     row_range=np.ptp(np.ptp(img.numpy(), axis=0)),
        #     column_range=np.ptp(np.ptp(img.numpy(), axis=1)),
        #     mean=np.mean(img.numpy()),
        #     shape=img.numpy().shape,
        # )
        # print("modified image", summary)
        if not self.predict:
            # create target mask
            target = np.zeros((image_shape, image_shape))
            # if patient ID has associated target boxes (=if image contains pneumonia)
            if pId in self.boxes:
                # loop through boxes
                # print("boxes:", pId, self.boxes[pId])
                for box in self.boxes[pId]:
                    # extract box coordinates
                    x, y, w, h = box
                    # rescale box coordinates
                    x = int(round(x / self.rescale_factor))
                    y = int(round(y / self.rescale_factor))
                    w = int(round(w / self.rescale_factor))
                    h = int(round(h / self.rescale_factor))
                    # create a mask of 1s (255 is used because pytorch will rescale to 0-1) inside the box
                    target[y : y + h, x : x + w] = 255  #
                    target[
                        target > 255
                    ] = 255  # correct in case of overlapping boxes (shouldn't happen)
            # add trailing channel dimension
            target = np.expand_dims(target, -1)
            target = target.astype("uint8")
            # summary = dict(
            #     row_range=np.ptp(np.ptp(target, axis=0)),
            #     column_range=np.ptp(np.ptp(target, axis=1)),
            #     mean=np.mean(target),
            #     shape=target.shape,
            # )
            # print("target:", pId, summary)
            # apply rotation augmentation
            if self.rotation_angle > 0:
                target = tv.transforms.functional.to_pil_image(target)
                target = tv.transforms.functional.rotate(
                    target, angle, resample=PIL.Image.BILINEAR
                )
            # apply transforms to target
            if self.transform is not None:
                target = self.transform(target)
            # summary = dict(
            #     row_range=np.ptp(np.ptp(target.numpy(), axis=0)),
            #     column_range=np.ptp(np.ptp(target.numpy(), axis=1)),
            #     mean=np.mean(target.numpy()),
            #     shape=target.numpy().shape,
            # )
            # print("target transformed:", pId, summary)
            return img, target, pId
        else:
            return img, pId

    def __len__(self):
        return len(self.pIds)
