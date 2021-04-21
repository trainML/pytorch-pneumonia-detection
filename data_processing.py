import os
import pandas as pd
import numpy as np
import pydicom


def get_boxes_per_patient(df, pId):
    """
    Given the dataset and one patient ID,
    return an array of all the bounding boxes and their labels associated with that patient ID.
    Example of return:
    array([[x1, y1, width1, height1, class1, target1],
           [x2, y2, width2, height2, class2, target2]])
    """

    boxes = df.loc[df["patientId"] == pId][
        ["x", "y", "width", "height", "class", "Target"]
    ].values
    return boxes


def get_metadata_from_dcm_file(file, attribute):
    """
    Given a patient ID, return useful meta-data from the corresponding dicom image header.
    Return:
    attribute value
    """
    # get dicom image
    dcmdata = pydicom.read_file(file)
    # extract attribute values
    attribute_value = getattr(dcmdata, attribute)
    return attribute_value


def build_feature_csv(images_path, output_file, boxes=None, class_info=None):
    patientIDs = [f[:-4] for f in os.listdir(images_path)]
    df = pd.DataFrame(data={"patientId": patientIDs})
    attributes = ["PatientSex", "PatientAge", "ViewPosition"]
    for a in attributes:
        df[a] = df["patientId"].apply(
            lambda x: get_metadata_from_dcm_file(f"{images_path}/{x}.dcm", a)
        )
    # convert patient age from string to numeric
    df["PatientAge"] = df["PatientAge"].apply(pd.to_numeric, errors="coerce")
    # remove a few outliers
    df["PatientAge"] = df["PatientAge"].apply(
        lambda x: x if x < 120 else np.nan
    )
    if boxes:
        df = df.join(boxes, on="patientId", how="left")
    if class_info:
        df = df.join(class_info, on="patientId", how="left")


def build_training_csv(images_path, labels_path, output_path):
    boxes = pd.read_csv(labels_path + "/stage_2_train_labels.csv")
    class_info = pd.read_csv(labels_path + "/stage_2_detailed_class_info.csv")
    class_info.drop_duplicates(inplace=True)
    build_feature_csv(
        images_path, f"{output_path}/train.csv", boxes, class_info
    )


def build_prediction_csv(images_path, output_path):
    build_feature_csv(
        images_path,
        f"{output_path}/predict.csv",
    )
