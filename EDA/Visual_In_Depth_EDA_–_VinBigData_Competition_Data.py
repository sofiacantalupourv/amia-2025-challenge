#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align: center; font-family: Verdana; font-size: 32px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; font-variant: small-caps; letter-spacing: 3px; color: #7b4f88; background-color: #ffffff;">VinBigData Chest X-ray Abnormalities Detection</h1>
# <h2 style="text-align: center; font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: underline; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">Exploratory Data Analysis (EDA)</h2>
# <h5 style="text-align: center; font-family: Verdana; font-size: 12px; font-style: normal; font-weight: bold; text-decoration: None; text-transform: none; letter-spacing: 1px; color: black; background-color: #ffffff;">CREATED BY: DARIEN SCHETTLER</h5>
# 

# <h2 style="font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; color: navy; background-color: #ffffff;">TABLE OF CONTENTS</h2>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;"><a href="#imports">0&nbsp;&nbsp;&nbsp;&nbsp;IMPORTS</a></h3>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;"><a href="#background_information">1&nbsp;&nbsp;&nbsp;&nbsp;BACKGROUND INFORMATION</a></h3>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;"><a href="#setup">2&nbsp;&nbsp;&nbsp;&nbsp;SETUP</a></h3>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;"><a href="#helper_functions">3&nbsp;&nbsp;&nbsp;&nbsp;HELPER FUNCTIONS</a></h3>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;"><a href="#tabular_data">4&nbsp;&nbsp;&nbsp;&nbsp;TABULAR DATA</a></h3>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;"><a href="#image_data">5&nbsp;&nbsp;&nbsp;&nbsp;IMAGE DATA</a></h3>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;"><a href="#combining_annotations">6&nbsp;&nbsp;&nbsp;&nbsp;COMBINING ANNOTATIONS</a></h3>
# 
# ---
# 

# <a style="text-align: font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; background-color: #ffffff; color: navy;" id="imports">0&nbsp;&nbsp;IMPORTS</a>

# In[1]:


# Machine Learning and Data Science Imports
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import tensorflow_hub as hub
from skimage import exposure
import pandas as pd; pd.options.mode.chained_assignment = None
import numpy as np
import scipy

# Built In Imports
from datetime import datetime
from glob import glob
import warnings
import IPython
import urllib
import zipfile
import pickle
import shutil
import string
import math
import tqdm
import time
import os
import gc
import re

# Visualization Imports
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from PIL import Image
import matplotlib
import plotly
import PIL
import cv2

# PRESETS
FIG_FONT = dict(family="Helvetica, Arial", size=14, color="#7f7f7f")
LABEL_COLORS = [px.colors.label_rgb(px.colors.convert_to_RGB_255(x)) for x in sns.color_palette("Spectral", 15)]
LABEL_COLORS_WOUT_NO_FINDING = LABEL_COLORS[:8]+LABEL_COLORS[9:]

# Other Imports
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm.notebook import tqdm
import pydicom

print("\n... IMPORTS COMPLETE ...\n")


# <a style="text-align: font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; color: navy; background-color: #ffffff;" id="background_information">1&nbsp;&nbsp;BACKGROUND INFORMATION</a>

# <h3 style="text-align: font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">1.1  THE DATA</h3>
# 
# ---
# 
# <b style="text-decoration: underline; font-family: Verdana;">BACKGROUND INFORMATION</b>
# 
# In this competition, we are classifying common thoracic lung diseases and localizing critical findings. <br>**This is an object detection and classification problem.**
# 
# For each test image, you will be predicting a bounding box and class for all findings. If you predict that there are no findings, you should create a prediction of **`14 1 0 0 1 1`** *(14 is the class ID for no finding, and this provides a one-pixel bounding box with a confidence of 1.0)*
# 
# Note that the images are in **DICOM** format, which means they contain additional data that might be useful for visualizing and classifying.
# 
# ![Example Radiographs](https://i.imgur.com/QWmbhXx.png)
# 
# <br>
# 
# <b style="text-decoration: underline; font-family: Verdana;">DATASET INFORMATION</b>
# 
# The dataset comprises **`18,000`** postero-anterior (PA) CXR scans in DICOM format, which were de-identified to protect patient privacy.
# 
# All images were labeled by a panel of experienced radiologists for the presence of **14** critical radiographic findings as listed below:
# 
# > **`0`** - Aortic enlargement <br>
# **`1`** - Atelectasis <br>
# **`2`** - Calcification <br>
# **`3`** - Cardiomegaly <br>
# **`4`** - Consolidation <br>
# **`5`** - ILD <br>
# **`6`** - Infiltration <br>
# **`7`** - Lung Opacity <br>
# **`8`** - Nodule/Mass <br>
# **`9`** - Other lesion <br>
# **`10`** - Pleural effusion <br>
# **`11`** - Pleural thickening <br>
# **`12`** - Pneumothorax <br>
# **`13`** - Pulmonary fibrosis <br>
# **`14`** - "No finding" observation was intended to capture the absence of all findings above
# 
# Note that a key part of this competition is working with ground truth from multiple radiologists. That means that the same image will have multiple ground-truth labels as annotated by different radiologists.
# 
# <br>
# 
# <b style="text-decoration: underline; font-family: Verdana;">DATA FILES</b>
# > **`train.csv`** - the train set metadata, with one row for each object, including a class and a bounding box (multiple rows per image possible)<br>
# **`sample_submission.csv`** - a sample submission file in the correct format
# 
# <br>
# 
# <b style="text-decoration: underline; font-family: Verdana;">TRAIN COLUMNS</b>
# > **`image_id`** - unique image identifier<br>
# **`class_name`** - the name of the class of detected object (or "No finding")<br>
# **`class_id`** - the ID of the class of detected object<br>
# **`rad_id`** - the ID of the radiologist that made the observation<br>
# **`x_min`** - minimum X coordinate of the object's bounding box<br>
# **`y_min`** - minimum Y coordinate of the object's bounding box<br>
# **`x_max`** - maximum X coordinate of the object's bounding box<br>
# **`y_max`** - maximum Y coordinate of the object's bounding box

# <h3 style="text-align: font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">1.2  THE GOAL</h3>
# 
# ---
# 
# In this competition, you’ll automatically localize and classify **`14`** types of thoracic abnormalities from chest radiographs. You'll work with a dataset consisting of **`18,000`** scans that have been annotated by experienced radiologists. You can train your model with **`15,000`** independently-labeled images and will be evaluated on a test set of **`3,000`** images. These annotations were collected via VinBigData's web-based platform, VinLab. Details on building the dataset can be found in our recent paper “VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations”.

# <h3 style="text-align: font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">1.3  ADDITIONAL INFORMATION ON ABNORMALITIES</h3>
# 
# <p style="font-size: 10px; color: red; font-weight: bold; font-family: Verdana; text-transform: uppercase;">Much Of The Content For This Markdown Cell Comes From <a href="https://www.kaggle.com/sakuraandblackcat/chest-x-ray-knowledges-for-the-14-abnormalities">This Notebook</a> Written By The Talented <a href="https://www.kaggle.com/sakuraandblackcat">User ANZ</a></p>
# 
# ---
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">Aortic enlargement</b>
# * Aortic enlargement is known as a sign of an aortic aneurysm. This condition often occurs in the ascending aorta.
# * In general, the term aneurysm is used when the axial diameter is >5.0 cm for the ascending aorta and >4.0 cm for the descending aorta.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">Atelectasis</b>
# * Atelectasis is a condition where there is no air in part or all of the lungs and they have collapsed.
# * A common cause of atelectasis is obstruction of the bronchi.
# * In atelectasis, there is an increase in density on chest x-ray (usually whiter; black on black-and-white inversion images).
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">Calcification</b>
# * Calcium (calcification) may be deposited in areas where previous inflammation of the lungs or pleura has healed.
# * Many diseases or conditions can cause calcification on chest x-ray.
# * Calcification may occur in the Aorta (as with atherosclerosis) or it may occur in mediastinal lymph nodes (as with previous infection, tuberculosis, or histoplasmosis).
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">Cardiomegaly</b>
# * Cardiomegaly is usually diagnosed when the ratio of the heart's width to the width of the chest is more than 50%. This diagnostic criterion may be an essential basis for this competition.
# * Cardiomegaly can be caused by many conditions, including hypertension, coronary artery disease, infections, inherited disorders, and cardiomyopathies.
# * The heart-to-lung ratio criterion for the diagnosis of cardiomegaly is a ratio of greater than 0.5. However, this is only valid if the XRay is performed while the patient is standing. If the patient is sitting or in bed, this criterion cannot be used. To determine whether a patient is sitting or standing (and consequently whether this criteron is valid), we will detect the presence of air in the stomach (if there is no air in it, the patient is not standing and the criterion cannot be used)
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">Consolidation</b>
# * Consolidation is a decrease in lung permeability due to infiltration of fluid, cells, or tissue replacing the air-containing spaces in the alveoli.
# * Consolidation is officially referred to as air space consolidation.
# * On X-rays displaying air space consolidation, the lung field's density is increased, and pulmonary blood vessels are not seen, but black bronchi can be seen in the white background, which is called <i>"air bronchogram"</i>. Since air remains in the bronchial tubes, they do not absorb X-rays and appear black, and the black and white are reversed from normal lung fields.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">ILD</b>
# * ILD stands for <i>"Interstitial Lung Disease"</i>.
# * Interstitial Lung Disease is a general term for many conditions in which the interstitial space is injured.
# * The interstitial space refers to the walls of the alveoli (air sacs in the lungs) and the space around the blood vessels and small airways.
# * Chest radiographic findings include ground-glass opacities (i.e., an area of hazy opacification), linear reticular shadows, and granular shadows.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">Infiltration</b>
# * The infiltration of some fluid component into the alveoli causes an infiltrative shadow (Infiltration).
# * It is difficult to distinguish from consolidation and, in some cases, impossible to distinguish. Please see [this link](https://allnurses.com/consolidation-vs-infiltrate-vs-opacity-t483538/) for more information.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">Lung Opacity</b>
# * Lung opacity is a loose term with many potential interpretations/meanings. Please see this [kaggle discussion](https://www.kaggle.com/zahaviguy/what-are-lung-opacities) for more information.
# * Lung opacity can often be identified as any area in the chest radiograph that is <b>more white than it should be.</b>
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">Nodule/Mass</b>
# * Nodules and masses are seen primarily in lung cancer, and metastasis from other parts of the body such as colon cancer and kidney cancer, tuberculosis, pulmonary mycosis, non-tuberculous mycobacterium, obsolete pneumonia, and benign tumors.
# * A nodule/mass is a round shade (typically less than 3 cm in diameter – resulting in much smaller than average bounding boxes) that appears on a chest X-ray image.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">Other lesion</b>
# * Others include all abnormalities that do not fall into any other category. This includes bone penetrating images, fractures, subcutaneous emphysema, etc.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">Pleural effusion</b>
# * Pleural effusion is the accumulation of water outside the lungs in the chest cavity.
# * The outside of the lungs is covered by a thin membrane consisting of two layers known as the pleura. Fluid accumulation between these two layers (chest-wall/parietal-pleura and the lung-tissue/visceral-pleura) is called pleural effusion.
# * The findings of pleural effusion vary widely and vary depending on whether the radiograph is taken in the upright or supine position.
# * The most common presentation of pleural effusion is <b>elevation of the diaphragm on one side, flattening the diaphragm, or blunting the angle between rib and diaphragm (typically more than 30 degrees)</b>
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">Pleural thickening</b>
# * The pleura is the membrane that covers the lungs, and the change in the thickness of the pleura is called pleural thickening.
# * It is often seen in the uppermost part of the lung field (the apex of the lung).
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">Pneumothorax</b>
# * A pneumothorax is a condition in which air leaks from the lungs and accumulates in the chest cavity.
# * When air leaks and accumulates in the chest, it cannot expand outward like a balloon due to the ribs' presence. Instead, the lungs are pushed by the air and become smaller. In other words, a pneumothorax is a situation where air leaks from the lungs and the lungs become smaller (collapsed).
# * In a chest radiograph of a pneumothorax, the collapsed lung is whiter than normal, and the area where the lung is gone is uniformly black. Besides, the edges of the lung may appear linear.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">Pulmonary fibrosis</b>
# * Pulmonary Fibrosis is inflammation of the lung interstitium due to various causes, resulting in thickening and hardening of the walls, fibrosis, and scarring.
# * The fibrotic areas lose their air content, which often results in dense cord shadows or granular shadows.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">No finding</b>
# * There are no findings on x-ray images. This is the normal image and is the baseline image needed to differentiate from the abnormal image.

# <a style="font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; color: navy; background-color: #ffffff;" id="setup">2&nbsp;&nbsp;NOTEBOOK SETUP</a>

# In[2]:


# Define the root data directory
DATA_DIR = "/data/mhedas/common/challenge_dataset"

# Define the paths to the training and testing dicom folders respectively
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Capture all the relevant full train/test paths
TRAIN_DICOM_PATHS = [os.path.join(TRAIN_DIR, f_name) for f_name in os.listdir(TRAIN_DIR)]
TEST_DICOM_PATHS = [os.path.join(TEST_DIR, f_name) for f_name in os.listdir(TEST_DIR)]
print(f"\n... The number of training files is {len(TRAIN_DICOM_PATHS)} ...")
print(f"... The number of testing files is {len(TEST_DICOM_PATHS)} ...")

# Define paths to the relevant csv files
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SS_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

# Create the relevant dataframe objects
train_df = pd.read_csv(TRAIN_CSV)
ss_df = pd.read_csv(SS_CSV)

print("\n\nTRAIN DATAFRAME\n\n")
display(train_df.head(3))

print("\n\nSAMPLE SUBMISSION DATAFRAME\n\n")
display(ss_df.head(3))


# <a style="text-align: font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; color: navy; background-color: #ffffff;" id="helper_functions">3&nbsp;&nbsp;HELPER FUNCTIONS</a>

# In[3]:


def dicom2array(path, voi_lut=True, fix_monochrome=True):
    """ Convert dicom file to numpy array

    Args:
        path (str): Path to the dicom file to be converted
        voi_lut (bool): Whether or not VOI LUT is available
        fix_monochrome (bool): Whether or not to apply monochrome fix

    Returns:
        Numpy array of the respective dicom file

    """
    # Use the pydicom library to read the dicom file
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to
    # transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # The XRAY may look inverted
    #   - If we want to fix this we can
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    # Normalize the image array and return
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

def plot_image(img, title="", figsize=(8,8), cmap=None):
    """ Function to plot an image to save a bit of time """
    plt.figure(figsize=figsize)

    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        img
        plt.imshow(img)

    plt.title(title, fontweight="bold")
    plt.axis(False)
    plt.show()

def get_image_id(path):
    """ Function to return the image-id from a path """
    return path.rsplit("/", 1)[1].rsplit(".", 1)[0]

def create_fractional_bbox_coordinates(row):
    """ Function to return bbox coordiantes as fractions from DF row """
    frac_x_min = row["x_min"]/row["img_width"]
    frac_x_max = row["x_max"]/row["img_width"]
    frac_y_min = row["y_min"]/row["img_height"]
    frac_y_max = row["y_max"]/row["img_height"]
    return frac_x_min, frac_x_max, frac_y_min, frac_y_max

def draw_bboxes(img, tl, br, rgb, label="", label_location="tl", opacity=0.1, line_thickness=0):
    """ TBD

    Args:
        TBD

    Returns:
        TBD
    """
    rect = np.uint8(np.ones((br[1]-tl[1], br[0]-tl[0], 3))*rgb)
    sub_combo = cv2.addWeighted(img[tl[1]:br[1],tl[0]:br[0],:], 1-opacity, rect, opacity, 1.0)
    img[tl[1]:br[1],tl[0]:br[0],:] = sub_combo

    if line_thickness>0:
        img = cv2.rectangle(img, tuple(tl), tuple(br), rgb, line_thickness)

    if label:
        # DEFAULTS
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 1.666
        FONT_THICKNESS = 3
        FONT_LINE_TYPE = cv2.LINE_AA

        if type(label)==str:
            LABEL = label.upper().replace(" ", "_")
        else:
            LABEL = f"CLASS_{label:02}"

        text_width, text_height = cv2.getTextSize(LABEL, FONT, FONT_SCALE, FONT_THICKNESS)[0]

        label_origin = {"tl":tl, "br":br, "tr":(br[0],tl[1]), "bl":(tl[0],br[1])}[label_location]
        label_offset = {
            "tl":np.array([0, -10]), "br":np.array([-text_width, text_height+10]),
            "tr":np.array([-text_width, -10]), "bl":np.array([0, text_height+10])
        }[label_location]
        img = cv2.putText(img, LABEL, tuple(label_origin+label_offset),
                          FONT, FONT_SCALE, rgb, FONT_THICKNESS, FONT_LINE_TYPE)

    return img


# <a style="text-align: font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; color: navy; background-color: #ffffff;" id="tabular_data">4&nbsp;&nbsp;TABULAR DATA</a>
# 
# <b style="text-decoration: underline; font-family: Verdana;">RECALL THAT THESE ARE THE TRAIN COLUMNS</b>
# > **`image_id`** - unique image identifier<br>
# **`class_name`** - the name of the class of detected object (or "No finding")<br>
# **`class_id`** - the ID of the class of detected object<br>
# **`rad_id`** - the ID of the radiologist that made the observation<br>
# **`x_min`** - minimum X coordinate of the object's bounding box<br>
# **`y_min`** - minimum Y coordinate of the object's bounding box<br>
# **`x_max`** - maximum X coordinate of the object's bounding box<br>
# **`y_max`** - maximum Y coordinate of the object's bounding box

# <h3 style="text-align: font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">4.1  IMAGE_ID COLUMN EXPLORATION</h3>
# 
# ---
# 
# The **`image_id`** column contains a **U**nique **ID**entifier (**UID**) that <b style="text-decoration: underline;">indicates which patient the respective row (object) relates to</b>.
# 
# As there can be up to three radiologists annotating the same image and potentially multiple objects/bboxes per image, it is possible for a single image UID to occur many times. However, please note that we know from the competition data details that there exists ***only one image for one patient***. This means that if a specific image_id appears 12 times, that there are 4 objects in the image, and each object was annotated by all three radiologists.
# 
# *SIDE-NOTE* – Due to the ***one image to one patient*** rule, the column name **`image_id`** could be replaced with **`patient_id`** and it would mean exactly the same thing.
# 
# <br><br>
# 
# <b style="text-decoration: underline; font-family: Verdana;">TOTAL OBJECT ANNOTATIONS PER IMAGE</b>
# 
# Let's count the distribution of the amount of annotations per unique **`image_id`** value. Note that we use a log-axis for the count axis to handle the large number of values present at 3 annotations (a single object annotated similarily by 3 radiologists)
# 
# ---
# 
# **From the histogram plotted below we can ascertain the following information:**
# * Images contain at least 3 annotations (1 distinct object annotation by 3 radiologists)
# * Images contain at most 57 annotations (19 distinct object annotations by 3 radiologists)
# * The vast majority of images only have 3 annotations (~11,000 out of 15,000 images)
# * The distribution has a heavy skew (**`value=3.8687`** **`# FROM --> scipy.stats.skew(train_df.image_id.value_counts().values)`**). Remember that a perfectly symetrical distribution would have a skew value of **`0`**.

# In[4]:


fig = px.histogram(train_df.image_id.value_counts(),
                   log_y=True, color_discrete_sequence=['indianred'], opacity=0.7,
                   labels={"value":"Number of Annotations Per Image"},
                   title="<b>DISTRIBUTION OF # OF ANNOTATIONS PER PATIENT   " \
                         "<i><sub>(Log Scale for Y-Axis)</sub></i></b>",
                   )
fig.update_layout(showlegend=False,
                  xaxis_title="<b>Number of Unique Images</b>",
                  yaxis_title="<b>Count of All Object Annotations</b>",
                  font=FIG_FONT,)
fig.show()


# <b style="text-decoration: underline; font-family: Verdana;">UNIQUE OBJECT ANNOTATIONS PER IMAGE</b>
# 
# Let's count the distribution of **UNIQUE** object-label annotations per unique **`image_id`** value. This means if a radiologist identifies 8 nodules in an image, we count that as 1 unique object annotation. The goal of this is to determine the distributions of different diseases occuring within the same patient.
# 
# Note that we use a log-axis for the count axis to handle the large number of values present at 1 unique abnormality
# 
# ---
# 
# **From the histogram plotted below we can ascertain the following information:**
# * Images contain no more than 10 unique abnormalities (out of a possible 14)
# * The more unique abnormalities present in an image, the rarer it is.

# In[5]:


fig = px.histogram(train_df.groupby('image_id')["class_name"].unique().apply(lambda x: len(x)),
             log_y=True, color_discrete_sequence=['skyblue'], opacity=0.7,
             labels={"value":"Number of Unique Abnormalities"},
             title="<b>DISTRIBUTION OF # OF ANNOTATIONS PER PATIENT   " \
                   "<i><sub>(Log Scale for Y-Axis)</sub></i></b>",
                   )
fig.update_layout(showlegend=False,
                  xaxis_title="<b>Number of Unique Abnormalities</b>",
                  yaxis_title="<b>Count of Unique Patients</b>",
                  font=FIG_FONT,)
fig.show()


# <h3 style="text-align: font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">4.2  CLASS_NAME COLUMN EXPLORATION</h3>
# 
# ---
# 
# The **`class_name`** column indicates the <b style="text-decoration: underline;">label as a string</b> for the respective object/annotation (each row is for one object/annotation).
# <br><br>
# 
# <b style="text-decoration: underline; font-family: Verdana;">ANNOTATIONS PER CLASS</b>
# 
# We know there are 15 different possible **`class_name`**s (including **`No finding`**). To identify the distribution of counts across the labels we will use a bar-chart.

# In[6]:


fig = px.bar(train_df.class_name.value_counts().sort_index(),
             color=train_df.class_name.value_counts().sort_index().index, opacity=0.85,
             color_discrete_sequence=LABEL_COLORS, log_y=True,
             labels={"y":"Annotations Per Class", "x":""},
             title="<b>Annotations Per Class</b>",)
fig.update_layout(legend_title=None,
                  font=FIG_FONT,
                  xaxis_title="",
                  yaxis_title="<b>Annotations Per Class</b>")

fig.show()


# <h3 style="text-align: font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">4.3  CLASS_ID COLUMN EXPLORATION</h3>
# 
# ---
# 
# The **`class_id`** column indicates the <b style="text-decoration: underline;">label encoded as a number</b> the respective object/annotation (each row is for one object/annotation). Knowing this, we will remove the previous **class_name** column, as we would rather work with a numeric representation. Prior to removal we will generate a map that will allow us to translate the numeric labels back into their respective string represntations.

# In[7]:


# Create dictionary mappings
int_2_str = {i:train_df[train_df["class_id"]==i].iloc[0]["class_name"] for i in range(15)}
str_2_int = {v:k for k,v in int_2_str.items()}
int_2_clr = {str_2_int[k]:LABEL_COLORS[i] for i,k in enumerate(sorted(str_2_int.keys()))}

print("\n... Dictionary Mapping Class Integer to Class String Representation [int_2_str]...\n")
display(int_2_str)

print("\n... Dictionary Mapping Class String to Class Integer Representation [str_2_int]...\n")
display(str_2_int)

print("\n... Dictionary Mapping Class Integer to Color Representation [str_2_clr]...\n")
display(int_2_clr)

print("\n... Head of Train Dataframe After Dropping The Class Name Column...\n")
train_df.drop(columns=["class_name"], inplace=True)
display(train_df.head(5))


# <h3 style="text-align: font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">4.4  RAD_ID COLUMN EXPLORATION</h3>
# 
# ---
# 
# The **`rad_id`** column indicates the <b style="text-decoration: underline;">the ID of the radiologist that made the observation</b>. Remember, three radiologists will annotate a given image out of a pool of seventeen possible radiologists, where the radiologist ID is encoded from R1 to R17.
# <br><br>
# 
# <b style="text-decoration: underline; font-family: Verdana;">ANNOTATIONS PER RADIOLOGIST</b>
# 
# We know there are 17 possible radiologists (**`rad_id`**s). To identify the distribution of annotations performed across the radiologists we will use a historgram.
# 
# ---
# 
# **From the histogram plotted below we can ascertain the following information**
# * 3 of the radiologists (R9, R10, & R8 in that order) are responsible for the vast majority of annotations (~40-50% of all annotations)
# * Among the other 14 radiologists there is some variation around the number of annotations made, however, these 14 radiologists all made between 3121 annotations and 812 annotations with the vast majority annotating 1800-2200 objects.

# In[8]:


fig = px.histogram(train_df, x="rad_id", color="rad_id",opacity=0.85,
                   labels={"rad_id":"Radiologist ID"},
                   title="<b>DISTRIBUTION OF # OF ANNOTATIONS PER RADIOLOGIST</b>",
                   ).update_xaxes(categoryorder="total descending")
fig.update_layout(legend_title="<b>RADIOLOGIST ID</b>",
                  xaxis_title="<b>Radiologist ID</b>",
                  yaxis_title="<b>Number of Annotations Made</b>",
                  font=FIG_FONT,)
fig.show()


# <b style="text-decoration: underline; font-family: Verdana;">ANNOTATIONS PER RADIOLOGIST SEPERATED BY CLASS LABEL</b>
# 
# We have already identified that three of the radiologists are responsible for almost 50% of all of the annotations. We would now like to identify if all of the radiologists were able to see and annotate all 15 classes. If so, can we identify any additional skew or problems that might arise?
# 
# ---
# 
# **From the first histogram plotted below we can ascertain the following information**
# * 3 of the radiologists (R9, R10, & R8 in that order) are responsible for the vast majority of annotations (~40-50% of all annotations)
# * Among the other 11 radiologists there is some variation around the number of annotations made, however, these 11 radiologists all made between 3121 annotations and 812 annotations with the vast majority annotating 1800-2200 objects.
# 
# ---
# 
# **From the second histogram plotted below we can ascertain the following information**
# * Among the other 11 radiologists, 7 of them (R1 through R7) have only ever annotated images as **`No finding`**
# * The other 4 radiologists are also heavily skewed towards the **`No finding`** label when compared to the main 3 radiologists (R8 through R10). This seems to actually be closer to the overall distribution, however it might allow us to estimate that radiologists other than R8, R9, and R10, are much more likely to annotate images as **`No finding`**.
# * The downside to this distribution, is that if we include this information in the model than the model will learn that 7 of the radiologists classify images as **`No finding`** 100% of the time!
# 
# <sup><b><i>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Note that this second plot could have been generated by interacting with the first histogram as plotly has this functionality built-in</i></b></sup>

# In[9]:


# #################################################################### #
#  TO DO - NORMALIZE RADIOLOGIST COUNTS BASED ON ANNOTATION PER IMAGE  #
# #################################################################### #

fig = go.Figure()

for i in range(15):
    fig.add_trace(go.Histogram(
        x=train_df[train_df["class_id"]==i]["rad_id"],
        marker_color=int_2_clr[i],
        name=f"<b>{int_2_str[i]}</b>"))

fig.update_xaxes(categoryorder="total descending")
fig.update_layout(title="<b>DISTRIBUTION OF CLASS LABEL ANNOTATIONS BY RADIOLOGIST</b>",
                  barmode='stack',
                  xaxis_title="<b>Radiologist ID</b>",
                  yaxis_title="<b>Number of Annotations Made</b>",
                  font=FIG_FONT,)
fig.show()

fig = go.Figure()
for i in range(15):
    fig.add_trace(go.Histogram(
        x=train_df[(train_df["class_id"]==i) & (~train_df["rad_id"].isin(["R8","R9","R10"]))]["rad_id"],
        marker_color=int_2_clr[i],
        name=f"<b>{int_2_str[i]}</b>"))

fig.update_xaxes(categoryorder="total descending")
fig.update_layout(title="<b>DISTRIBUTION OF CLASS LABEL ANNOTATIONS BY RADIOLOGIST   " \
                  "<i><sub>(EXCLUDING TOP 3 RADIOLOGISTS --> R8, R9 & R10)</sub></i></b>",
                  barmode='stack',
                  xaxis_title="<b>Radiologist ID</b>",
                  yaxis_title="<b>Number of Annotations Made</b>",
                  font=FIG_FONT,)
fig.show()


# <h3 style="text-align: font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">4.5  EXPLORATION OF BBOX COORDINATE COLUMNS</h3>
# 
# ---
# 
# The **`x_min`**, **`y_min`**, **`x_max`**, and **`y_max`** columns indicate the <b style="text-decoration: underline;">location of the annotated object bounding box</b>, where the top-left corner is represented by the tuple (**`x_min`**, **`y_min`**) and the bottom-right corner is represented by the tuple (**`x_max`**, **`y_max`**).
# 
# A value of **`NaN`** coincides with a label 14 (**`No finding`**) and means that there is nothing to annotate (healthy x-ray).<br>
# For the purpose of examining these columns we will <b style="text-decoration: underline;">only be examining rows where the objects have been annotated with a bounding box</b><br>
# (i.e. All rows with a label of **`No finding`** will be discarded)<br><br>
# 
# <b style="text-decoration: underline; font-family: Verdana;">PLOT HEATMAP REPRESENTING BOUNDING BOXES FOR VARIOUS CLASSES</b>
# 
# There's a lot to digest within these plots. The important thing to focus on will be identifying for each class the approximate range of locations the annotations are found in and the intensity of the locations within the heatmap.
# 
# ---
# 
# **From the heatmaps plotted below we can ascertain the following information**
# * Regarding Aortic Enlargement <i><sub>(CLASS-ID: 0)</sub></i>
#     * Heatmap distribution is slightly oval (vertical) and is very tight and intense, located in the centre of the image (slight drift to the top-right).
# * Regarding Atelectasis <i><sub>(CLASS-ID: 1)</sub></i>
#     * Heatmap distribution is lung shaped and relatively diffuse with a circular focus on the upper-left part of the left lung.
# * Regarding Calcification <i><sub>(CLASS-ID: 2)</sub></i>
#     * Heatmap distribution is lung shaped and relatively diffuse with a oval (vertical) focus on the top-left edge of the right lung.
# * Regarding Cardiomegaly <i><sub>(CLASS-ID: 3)</sub></i>
#     * Heatmap distribution is rectangular and is very tight and intense, located in the bottom-centre (to bottom-centre-right) of the image.
# * Regarding Consolidation <i><sub>(CLASS-ID: 4)</sub></i>
#     * Heatmap distribution is lung shaped and relatively diffuse, the focus of the distribution covers the entire left lung.
# * Regarding ILD <i><sub>(CLASS-ID: 5)</sub></i>
#     * Heatmap distribution is lung shaped and relatively diffuse, the focus leans a little towards the centre of the lungs.
# * Regarding Infiltration <i><sub>(CLASS-ID: 6)</sub></i>
#     * Heatmap distribution is lung shaped and relatively diffuse, the focus of the distribution covers the entire left lung.
# * Regarding Lung Opacity <i><sub>(CLASS-ID: 7)</sub></i>
#     * Heatmap distribution is lung shaped and relatively diffuse, the focus of the distribution covers the entire left lung.
# * Regarding Nodule/Mass <i><sub>(CLASS-ID: 8)</sub></i>
#     * Heatmap distribution is lung shaped and relatively diffuse, the focus leans a little towards the centre of the lungs. <b>(NOTE: The diffusion pattern looks patchy... probably due to smaller bounding boxes)</b>
# * Regarding Other Lesion <i><sub>(CLASS-ID: 9)</sub></i>
#     * Heatmap distribution is incredibly diffuse and covers most of the image, the focus is towards a vertical-strip in the centre of the image.
# * Regarding Pleural Effusion <i><sub>(CLASS-ID: 10)</sub></i>
#     * Heatmap distribution is lung shaped (slightly more rectangular?) and relatively diffuse, the focus is towards the bottom of the lungs and although both lungs are covered, the left lung has a stronger focus.
# * Regarding Pleural Thickening <i><sub>(CLASS-ID: 11)</sub></i>
#     * Heatmap distribution is vaguely lung shaped (patches near top and focus trails down exterior lung edge fading as it goes), the focus is towards the top of the lungs is oval (horizontal).
# * Regarding Pneumothorax <i><sub>(CLASS-ID: 12)</sub></i>
#     * Heatmap distribution is lung shaped (more rectangular), the focus is on the entire left lung however the right lung has some diffuse coverage.
# * Regarding Pulmonary Fibrosis <i><sub>(CLASS-ID: 13)</sub></i>
#     * Heatmap distribution is vaguely lung shaped (patches near top and focus trails down lung fading as it goes), the focus is towards the top of the lung and it is oval.
# 

# In[13]:


def png2array(path, normalize=True):
    """ Convert png file to numpy array
    Args:
        path (str): Path to the png file to be converted
        normalize (bool): Whether or not to normalize the image data
    Returns:
        Numpy array of the respective png file
    """
    # Use PIL/Pillow to read the png file
    image = Image.open(path)
    
    # Convert the image to a numpy array
    data = np.array(image)
    
    # Check if image has an alpha channel (4 channels) and remove it if present
    if len(data.shape) == 3 and data.shape[2] == 4:
        data = data[:, :, :3]  # Keep only RGB channels
    
    # Convert to grayscale if the image is RGB
    if len(data.shape) == 3 and data.shape[2] == 3:
        # Convert RGB to grayscale using standard formula
        data = np.dot(data[...,:3], [0.2989, 0.5870, 0.1140])
        data = data.astype(np.uint8)
    
    # Normalize the image if requested
    if normalize:
        data = data - np.min(data)
        if np.max(data) > 0:  # Avoid division by zero
            data = data / np.max(data)
            data = (data * 255).astype(np.uint8)
    
    return data


# In[73]:


def load_image_dimensions(csv_path):
    """Load image dimensions from CSV file
    
    Args:
        csv_path (str): Path to the CSV file with image dimensions
        
    Returns:
        dict: Dictionary mapping image_id to (height, width) tuple
    """
    img_dimensions = {}
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        # Store dimensions as (height, width) to be consistent with previous code
        img_dimensions[row['image_id']] = (int(row['dim0']), int(row['dim1']))
    
    return img_dimensions


# In[76]:


# Get paths to images where bboxes exist `class_id!=14`
bbox_df = train_df[train_df.class_id!=14].reset_index(drop=True)
BBOX_PATHS = [
    os.path.join(TRAIN_DIR, "train", name+".png") \
    for name in bbox_df.image_id.unique()
]

# Initialize our maps for current and original image sizes
current_image_sizes = {}  # Sizes of the PNG images (what we actually have)
original_image_sizes = {} # Original sizes from img_size.csv

# Load original dimensions from the CSV file
original_image_sizes_df = pd.read_csv('/data/mhedas/common/challenge_dataset/img_size.csv')
for _, row in original_image_sizes_df.iterrows():
    # Store as (height, width)
    original_image_sizes[row['image_id']] = (int(row['dim0']), int(row['dim1']))

# Get the current sizes of the PNG images
print("Getting current image sizes...")
for path in tqdm(BBOX_PATHS, total=len(BBOX_PATHS)):
    image_id = path[:-4].rsplit("/", 1)[1]
    image = Image.open(path)
    width, height = image.size  # PIL returns width, height
    current_image_sizes[image_id] = (height, width)  # Store as height, width to match original format
    image.close()  # Close the image to free up memory

# Add both current and original dimensions to the dataframe
bbox_df["current_height"] = bbox_df["image_id"].map(lambda x: current_image_sizes.get(x, (0, 0))[0])
bbox_df["current_width"] = bbox_df["image_id"].map(lambda x: current_image_sizes.get(x, (0, 0))[1])
bbox_df["original_height"] = bbox_df["image_id"].map(lambda x: original_image_sizes.get(x, (0, 0))[0])
bbox_df["original_width"] = bbox_df["image_id"].map(lambda x: original_image_sizes.get(x, (0, 0))[1])

# Create a function to scale the bounding boxes from original to current dimensions
def scale_bbox_coordinates(row):
    """Scale bounding box coordinates from original to current dimensions"""
    # Original dimensions
    orig_height = row['original_height']
    orig_width = row['original_width']
    
    # Current dimensions
    curr_height = row['current_height']
    curr_width = row['current_width']
    
    # Original bbox coordinates
    x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
    
    # Scale factors
    width_scale = curr_width / orig_width if orig_width > 0 else 1
    height_scale = curr_height / orig_height if orig_height > 0 else 1
    
    # Scale coordinates
    scaled_x_min = x_min * width_scale
    scaled_x_max = x_max * width_scale
    scaled_y_min = y_min * height_scale
    scaled_y_max = y_max * height_scale
    
    # Return scaled coordinates
    return scaled_x_min, scaled_x_max, scaled_y_min, scaled_y_max

# Apply scaling to get the scaled bbox coordinates
bbox_df["scaled_x_min"], bbox_df["scaled_x_max"], bbox_df["scaled_y_min"], bbox_df["scaled_y_max"] = \
    zip(*bbox_df.apply(scale_bbox_coordinates, axis=1))

# Now create the fractional coordinates based on the SCALED bounding boxes
def create_fractional_bbox_coordinates(row):
    """Create fractional bounding box coordinates (as a percentage of image dimensions)"""
    return (
        row['scaled_x_min'] / row['current_width'] if row['current_width'] > 0 else 0,
        row['scaled_x_max'] / row['current_width'] if row['current_width'] > 0 else 0,
        row['scaled_y_min'] / row['current_height'] if row['current_height'] > 0 else 0,
        row['scaled_y_max'] / row['current_height'] if row['current_height'] > 0 else 0
    )

# Apply the function to create fractional coordinates
bbox_df["frac_x_min"], bbox_df["frac_x_max"], bbox_df["frac_y_min"], bbox_df["frac_y_max"] = \
    zip(*bbox_df.apply(create_fractional_bbox_coordinates, axis=1))

# Calculate average dimensions of the current images for the heatmap
ave_src_img_height = np.mean([size[0] for size in current_image_sizes.values()], dtype=np.int32)
ave_src_img_width = np.mean([size[1] for size in current_image_sizes.values()], dtype=np.int32)

print(f"Average current image dimensions: {ave_src_img_height} × {ave_src_img_width}")
bbox_df.head()


# In[77]:


# DEFAULT
HEATMAP_SIZE = (ave_src_img_height, ave_src_img_width, 14)

# Initialize
heatmap = np.zeros((HEATMAP_SIZE), dtype=np.int16)

# Convert fractional coordinates to pixel coordinates for the heatmap
bbox_np = bbox_df[["class_id", "frac_x_min", "frac_x_max", "frac_y_min", "frac_y_max"]].to_numpy()
bbox_np[:, 1:3] *= ave_src_img_width   # Scale x coordinates
bbox_np[:, 3:5] *= ave_src_img_height  # Scale y coordinates
bbox_np = np.floor(bbox_np).astype(np.int16)

# Ensure coordinates are within heatmap bounds
bbox_np[:, 1] = np.clip(bbox_np[:, 1], 0, ave_src_img_width-1)
bbox_np[:, 2] = np.clip(bbox_np[:, 2], 0, ave_src_img_width-1)
bbox_np[:, 3] = np.clip(bbox_np[:, 3], 0, ave_src_img_height-1)
bbox_np[:, 4] = np.clip(bbox_np[:, 4], 0, ave_src_img_height-1)

# Color map stuff
custom_cmaps = [
    matplotlib.colors.LinearSegmentedColormap.from_list(
        colors=[(0.,0.,0.), c, (0.95,0.95,0.95)],
        name=f"custom_{i}") for i,c in enumerate(sns.color_palette("Spectral", 15))
]
custom_cmaps.pop(8)  # Remove No-Finding

# Fill the heatmap
print("Creating heatmap...")
for row in tqdm(bbox_np, total=bbox_np.shape[0]):
    # Skip invalid boxes
    if row[1] >= row[2] or row[3] >= row[4]:
        continue
    heatmap[row[3]:row[4]+1, row[1]:row[2]+1, row[0]] += 1

# Plot the heatmaps
fig = plt.figure(figsize=(20,25))
plt.suptitle("Heatmaps Showing Bounding Box Placement\n(Properly Scaled from Original to Current Dimensions)", 
             fontweight="bold", fontsize=16)

for i in range(15):
    plt.subplot(4, 4, i+1)
    if i==0:
        plt.imshow(heatmap.mean(axis=-1), cmap="bone")
        plt.title(f"Average of All Classes", fontweight="bold")
    else:
        plt.imshow(heatmap[:, :, i-1], cmap=custom_cmaps[i-1])
        plt.title(f"{int_2_str[i-1]} – ({i})", fontweight="bold")
    plt.axis(False)

fig.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()


# <b style="text-decoration: underline; font-family: Verdana;">INVESTIGATE THE SIZES OF BOUNDING BOXES AND THE IMPACT OF CLASS</b>
# 
# As we wish to examine the average, as well as the upper and lower limits for various class-based bounding box statistics, we will use a box plot to investigate. To make things easier to understand let us consider the following basic buckets.
# 
# <b><u>Bounding Box Area - Median</u></b>
# * Under   0.01 –– <b>Smallest</b>
# * 0.01 to 0.02 –– <b>Small</b>
# * 0.02 to 0.04 –– <b>Medium</b>
# * 0.04 to 0.06 –– <b>Large</b>
# * Above   0.06 –– <b>Largest</b>
# 
# <b><u>Bounding Box Area - Quartile Range</u></b>
# * Under     0.0075 –– <b>Smallest</b>
# * 0.0075 to 0.0125 –– <b>Small</b>
# * 0.0125 to 0.0250 –– <b>Medium</b>
# * 0.0250 to 0.0500 –– <b>Large</b>
# * Above     0.0500 –– <b>Largest</b>
# 
# ---
# 
# **From the boxplot plotted below we can ascertain the following information**
# * Regarding Aortic Enlargement Box Plot <i><sub>(CLASS-ID: 0)</sub></i>
#     * Median Value is <b>Small</b>  –––  Quartile Range is <b>Smallest</b>
# * Regarding Atelectasis Box Plot <i><sub>(CLASS-ID: 1)</sub></i>
#     * Median Value is <b>Medium</b>  –––  Quartile Range is <b>Large</b>
# * Regarding Calcification Box Plot <i><sub>(CLASS-ID: 2)</sub></i>
#     * Median Value is <b>Smallest</b>  –––  Quartile Range is <b>Medium</b>
# * Regarding Cardiomegaly Box Plot <i><sub>(CLASS-ID: 3)</sub></i>
#     * Median Value is <b>Large</b>  –––  Quartile Range is <b>Large</b>
# * Regarding Consolidation Box Plot <i><sub>(CLASS-ID: 4)</sub></i>
#     * Median Value is <b>Medium</b>  –––  Quartile Range is <b>Large</b>
# * Regarding ILD Box Plot <i><sub>(CLASS-ID: 5)</sub></i>
#     * Median Value is <b>Largest</b>  –––  Quartile Range is <b>Largest</b>
# * Regarding Infiltration Box Plot <i><sub>(CLASS-ID: 6)</sub></i>
#     * Median Value is <b>Medium</b>  –––  Quartile Range is <b>Large</b>
# * Regarding Lung Opacity Box Plot <i><sub>(CLASS-ID: 7)</sub></i>
#     * Median Value is <b>Medium</b>  –––  Quartile Range is <b>Large</b>
# * Regarding Nodule/Mass Box Plot <i><sub>(CLASS-ID: 8)</sub></i>
#     * Median Value is <b>Smallest</b>  –––  Quartile Range is <b>Smallest</b>
# * Regarding Other Lesion Box Plot <i><sub>(CLASS-ID: 9)</sub></i>
#     * Median Value is <b>Small</b>  –––  Quartile Range is <b>Large</b>
# * Regarding Pleural Effusion Box Plot <i><sub>(CLASS-ID: 10)</sub></i>
#     * Median Value is <b>Smallest</b>  –––  Quartile Range is <b>Large</b>
# * Regarding Pleural Thickening Box Plot <i><sub>(CLASS-ID: 11)</sub></i>
#     * Median Value is <b>Smallest</b>  –––  Quartile Range is <b>Smallest</b>
# * Regarding Pneumothorax Box Plot <i><sub>(CLASS-ID: 12)</sub></i>
#     * Median Value is <b>Largest</b>  –––  Quartile Range is <b>Largest</b>
# * Regarding Pulmonary Fibrosis Box Plot <i><sub>(CLASS-ID: 13)</sub></i>
#     * Median Value is <b>Small</b>  –––  Quartile Range is <b>Medium</b>
# 

# In[16]:


# Update bbox dataframe to make this easier
bbox_df["frac_bbox_area"] = (bbox_df["frac_x_max"]-bbox_df["frac_x_min"])*(bbox_df["frac_y_max"]-bbox_df["frac_y_min"])
bbox_df["class_id_as_str"] = bbox_df["class_id"].map(int_2_str)
display(bbox_df.head())

fig = px.box(bbox_df.sort_values(by="class_id_as_str"), x="class_id_as_str", y="frac_bbox_area", color="class_id_as_str",
             color_discrete_sequence=LABEL_COLORS_WOUT_NO_FINDING, notched=True,
             labels={"class_id_as_str":"Class Name", "frac_bbox_area":"BBox Area (%)"},
             title="<b>DISTRIBUTION OF BBOX AREAS AS % OF SOURCE IMAGE AREA   " \
                   "<i><sub>(Some Upper Outliers Excluded For Better Visualization)</sub></i></b>")

fig.update_layout(showlegend=True,
                  yaxis_range=[-0.025,0.4],
                  legend_title_text=None,
                  xaxis_title="",
                  yaxis_title="<b>Bounding Box Area %</b>",
                  font=FIG_FONT,)
fig.show()


# <b style="text-decoration: underline; font-family: Verdana;">INVESTIGATE THE ASPECT RATIO OF BOUNDING BOXES AND THE IMPACT OF CLASS</b>
# 
# We want to understand the average shape (wide-narrow, square, etc.) of the bouning-boxes associated with each class, and to do this we will use a bar chart with some pre-drawn lines.
# 
# ---
# 
# **From the bar chart plotted below we can ascertain the following information:**
# * The average size of bounding-boxes by class is usually close to square (usually on the horizontal rectangle size of square).
# * <b style="text-decoration: underline;">Cardiomegaly</b> has, on average, very thin, rectangular, <b style="text-decoration: underline;">horizontal boxes</b> (mean width is ~2.9x larger than mean height).
# * <b style="text-decoration: underline;">Pleural Thickening</b> has, on average, thin, rectangular, <b style="text-decoration: underline;">horizontal boxes</b> (mean width is ~1.9x larger than mean height).
# * <b style="text-decoration: underline;">ILD</b> has, on average, somewhat thin, rectangular, <b style="text-decoration: underline;"> vertical boxes</b> (mean height is ~1.6x larger than mean width)
# 

# In[17]:


# Aspect Ratio is Calculated as Width/Height
bbox_df["aspect_ratio"] = (bbox_df["x_max"]-bbox_df["x_min"])/(bbox_df["y_max"]-bbox_df["y_min"])

# Display average means for each class_id so we can examine the newly created Aspect Ratio Column
display(bbox_df.groupby("class_id").mean())

# Generate the bar plot
fig = px.bar(x=[int_2_str[x] for x in range(14)], y=bbox_df.groupby("class_id").mean()["aspect_ratio"],
             color=[int_2_str[x] for x in range(14)], opacity=0.85,
             color_discrete_sequence=LABEL_COLORS_WOUT_NO_FINDING,
             labels={"x":"Class Name", "y":"Aspect Ratio (W/H)"},
             title="<b>Aspect Ratios For Bounding Boxes By Class</b>",)
fig.update_layout(font=FIG_FONT,
                  yaxis_title="<b>Aspect Ratio (W/H)</b>",
                  xaxis_title=None,
                  legend_title_text=None)
fig.add_hline(y=1, line_width=2, line_dash="dot",
              annotation_font_size=10,
              annotation_text="<b>SQUARE ASPECT RATIO</b>",
              annotation_position="bottom left",
              annotation_font_color="black")
fig.add_hrect(y0=0, y1=0.5, line_width=0, fillcolor="red", opacity=0.125,
              annotation_text="<b>>2:1 VERTICAL RECTANGLE REGION</b>",
              annotation_position="bottom right",
              annotation_font_size=10,
              annotation_font_color="red")
fig.add_hrect(y0=2, y1=3.5, line_width=0, fillcolor="green", opacity=0.04,
              annotation_text="<b>>2:1 HORIZONTAL RECTANGLE REGION</b>",
              annotation_position="top right",
              annotation_font_size=10,
              annotation_font_color="green")
fig.show()


# <a style="text-align: font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; color: navy; background-color: #ffffff;" id="image_data">5&nbsp;&nbsp;IMAGE DATA</a>
# 
# Recall that the image data is stored in DICOM format and the annotations are stored in our **`train_df`** Dataframe.

# <h3 style="text-align: font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">5.1  CLASS TO HELP PARSE THE DATA </h3>
# 
# ---
# 
# Classes are very useful when we need to bind data to functionality. In this case, I have created a class (unwieldy as it may be currently in it's initial version) to help with that called **`TrainData`**.
# 
# I will get into the details of how the methods work at a later time... and for today I will simply generate the outputs using each method to show their functionality.

# <b style="text-decoration: underline; font-family: Verdana;">PLOT IMAGES FROM THE CORRESPONDING IMAGE IDS</b>
# 
# Summary to be done later
# 
# ---
# 
# More detail to come later

# In[42]:


def draw_bboxes(img, tl, br, rgb=(0, 255, 0), label=None, label_location='top', 
                opacity=0.2, line_thickness=1):
    """Draw bounding boxes on an image.
    
    Args:
        img (numpy.ndarray): The image to draw on
        tl (tuple): Top-left corner of the box (x, y)
        br (tuple): Bottom-right corner of the box (x, y)
        rgb (tuple): Color of the box in RGB format
        label (str): Label to display
        label_location (str): Where to place the label ('top', 'bottom')
        opacity (float): Opacity of the filled rectangle
        line_thickness (int): Thickness of the bounding box lines
    
    Returns:
        numpy.ndarray: Image with bounding box drawn
    """
    # Ensure img is not modified in-place (in case of error)
    img_copy = img.copy()
    
    # Get image dimensions
    height, width = img_copy.shape[:2]
    
    # Make sure coordinates are within image boundaries
    tl = (max(0, min(tl[0], width-1)), max(0, min(tl[1], height-1)))
    br = (max(tl[0]+1, min(br[0], width)), max(tl[1]+1, min(br[1], height)))
    
    # Draw the rectangle outline
    cv2.rectangle(img_copy, tl, br, rgb, line_thickness)
    
    # Create filled rectangle with opacity
    if opacity > 0:
        box_height = br[1] - tl[1]
        box_width = br[0] - tl[0]
        
        # Skip if box dimensions are invalid
        if box_height <= 0 or box_width <= 0:
            return img_copy
            
        rect = np.uint8(np.ones((box_height, box_width, 3)) * np.array(rgb, dtype=np.uint8))
        try:
            sub_img = img_copy[tl[1]:br[1], tl[0]:br[0], :]
            sub_combo = cv2.addWeighted(sub_img, 1-opacity, rect, opacity, 1.0)
            img_copy[tl[1]:br[1], tl[0]:br[0], :] = sub_combo
        except Exception as e:
            print(f"Error in addWeighted: {e}, box: {tl}-{br}, img shape: {img_copy.shape}, rect shape: {rect.shape}")
    
    # Add label if provided
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        if label_location == 'top':
            text_position = (tl[0], max(0, tl[1] - 5))
        else:  # bottom
            text_position = (tl[0], min(height, br[1] + text_size[1] + 5))
            
        cv2.putText(img_copy, label, text_position, font, font_scale, rgb, font_thickness)
    
    return img_copy


# In[62]:


def scale_bbox(bbox, original_dims, new_dims):
    """
    Scale bounding box coordinates from original image dimensions to new dimensions
    
    Args:
        bbox (list/array): Bounding box coordinates [x_min, y_min, x_max, y_max]
        original_dims (tuple): Original image dimensions (height, width)
        new_dims (tuple): New image dimensions (height, width)
        
    Returns:
        list: Scaled bounding box coordinates [x_min, y_min, x_max, y_max]
    """
    orig_height, orig_width = original_dims
    new_height, new_width = new_dims
    
    # Scale factors
    width_scale = new_width / orig_width
    height_scale = new_height / orig_height
    
    # Scale coordinates
    x_min = int(bbox[0] * width_scale)
    y_min = int(bbox[1] * height_scale)
    x_max = int(bbox[2] * width_scale)
    y_max = int(bbox[3] * height_scale)
    
    return [x_min, y_min, x_max, y_max]


# In[63]:


class TrainData():
    def __init__(self, df, train_dir, original_dimensions, cmap="Spectral"):
        # Initialize
        self.df = df
        self.train_dir = train_dir
        self.original_dimensions = original_dimensions  # Dict mapping image_id to original (height, width)
        
        # Visualization
        self.cmap = cmap
        self.pal = [tuple([int(x) for x in np.array(c)*(255,255,255)]) for c in sns.color_palette(cmap, 15)]
        self.pal.pop(8)
        
        self.img_annotations = self.get_annotations(get_all=True)
        
    def get_annotations(self, get_all=False, image_ids=None, class_ids=None, rad_ids=None, index=None):
        """ Get image annotations based on various filters
        
        Args:
            get_all (bool, optional): Get all annotations
            image_ids (list of strs, optional): Filter by image IDs
            class_ids (list of ints, optional): Filter by class IDs
            rad_ids (list of strs, optional): Filter by radiologist IDs
            index (int, optional): Filter by index

        Returns:
            dict: Dictionary of annotations by image_id
        """
        if not get_all and image_ids is None and class_ids is None and rad_ids is None and index is None:
            raise ValueError("Expected one of the following arguments to be passed:" \
                             "\n\t\t– `get_all`, `image_id`, `class_id`, `rad_id`, or `index`")
        # Initialize
        tmp_df = self.df.copy()

        if not get_all:
            if image_ids is not None:
                tmp_df = tmp_df[tmp_df.image_id.isin(image_ids)]
            if class_ids is not None:
                tmp_df = tmp_df[tmp_df.class_id.isin(class_ids)]
            if rad_ids is not None:
                tmp_df = tmp_df[tmp_df.rad_id.isin(rad_ids)]
            if index is not None:
                tmp_df = tmp_df.iloc[index]

        annotations = {image_id:[] for image_id in tmp_df.image_id.to_list()}
        for row in tmp_df.to_numpy():
            # Update annotations dictionary
            annotations[row[0]].append(dict(
                img_path=os.path.join(self.train_dir, "train", row[0]+".png"),
                image_id=row[0],
                class_id=int(row[1]),
                rad_id=int(row[2][1:]),
            ))

            # Catch to convert float array to integer array
            if row[1]==14:
                annotations[row[0]][-1]["bbox"]=row[3:]
            else:
                annotations[row[0]][-1]["bbox"]=row[3:].astype(np.int32)
        return annotations
        
    def get_annotated_image(self, image_id, annots=None, plot=False, plot_size=(18,25), plot_title=""):
        if annots is None:
            annots = self.img_annotations.copy()

        if type(annots) != list:
            image_annots = annots[image_id]
        else:
            image_annots = annots

        # Load image
        img_path = image_annots[0]["img_path"]
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Get current image dimensions
        current_dims = img_array.shape[:2]  # (height, width)
        
        # Get original dimensions for this image
        original_dims = self.original_dimensions.get(image_id)
        
        if original_dims:
            # Draw annotations
            for ann in image_annots:
                if ann["class_id"] != 14:
                    bbox = ann["bbox"]
                    
                    # Scale bbox from original dimensions to current dimensions
                    scaled_bbox = scale_bbox(bbox, original_dims, current_dims)
                    
                    tl = (scaled_bbox[0], scaled_bbox[1])
                    br = (scaled_bbox[2], scaled_bbox[3])
                    
                    # Draw the bounding box
                    img_array = draw_bboxes(img_array,
                                tl, br,
                                rgb=self.pal[ann["class_id"]],
                                label=int_2_str[ann["class_id"]],
                                opacity=0.08, line_thickness=4)
        
        if plot:
            plot_image(img_array, title=plot_title, figsize=plot_size)

        return img_array
    
    def plot_image_ids(self, image_id_list, height_multiplier=6, verbose=True):
        annotations = self.get_annotations(image_ids=image_id_list)
        n = len(image_id_list)

        plt.figure(figsize=(20, height_multiplier*n))
        for i, (image_id, annots) in enumerate(annotations.items()):
            if i >= n:
                break
            if verbose:
                print(f".", end="")
            plt.subplot(n//2,2,i+1)
            try:
                img = self.get_annotated_image(image_id, annots)
                plt.imshow(img)
                plt.axis(False)
                plt.title(f"Image ID – {image_id}")
            except Exception as e:
                print(f"\nError processing image {image_id}: {e}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

    def plot_classes(self, class_list, n=4, height_multiplier=6, verbose=True):
        annotations = self.get_annotations(class_ids=class_list)
        annotated_imgs = []

        plt.figure(figsize=(20, height_multiplier*n))
        for i, (image_id, annots) in enumerate(annotations.items()):
            if i >= n:
                break
            if verbose:
                print(f".", end="")
            plt.subplot(n//2,2,i+1)
            plt.imshow(self.get_annotated_image(image_id, annots))
            plt.axis(False)
            plt.title(f"Image ID – {image_id}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

    def plot_radiologists(self, rad_id_list, n=4, height_multiplier=6, verbose=True):
        annotations = self.get_annotations(rad_ids=rad_id_list)
        annotated_imgs = []

        plt.figure(figsize=(20, height_multiplier*n))
        for i, (image_id, annots) in enumerate(annotations.items()):
            if i >= n:
                break
            if verbose:
                print(f".", end="")
            plt.subplot(n//2,2,i+1)
            plt.imshow(self.get_annotated_image(image_id, annots))
            plt.axis(False)
            plt.title(f"Image ID – {image_id}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()


# In[65]:


img_dimensions = load_image_dimensions('/data/mhedas/common/challenge_dataset/img_size.csv')
train_data = TrainData(train_df, TRAIN_DIR, original_dimensions=img_dimensions)


# In[58]:


img = Image.open("/data/mhedas/common/challenge_dataset/train/train/x5N0ghWPbDxepww8hv4KT5AWIIxHZb6g.png")
img_array = np.array(img)
img.size


# In[60]:


img_dimensions["x5N0ghWPbDxepww8hv4KT5AWIIxHZb6g"]


# In[66]:


IMAGE_ID_LIST = train_df[train_df.class_id!=14].image_id[25:29].to_list()
train_data.plot_image_ids(image_id_list=IMAGE_ID_LIST, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">PLOT IMAGES CONTAINING A SINGLE CLASS</b>
# 
# Summary to be done later...
# 
# **NOTE: Only the bounding boxes for the specified classes will be drawn... probably a TBD in the future as a possible arg**
# 
# ---
# 
# More detail to come later

# In[67]:


train_data.plot_classes(class_list=[7,], n=2, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">PLOT IMAGES CONTAINING ONE OR MORE CLASSES FROM A LIST</b>
# 
# Summary to be done later...
# 
# **NOTE: Images need not contain ALL the classes... potential future improvement or option.**
# 
# ---
# 
# More detail to come later

# In[68]:


train_data.plot_classes(class_list=[5,8,11], n=4, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">PLOT IMAGES ANNOTATED BY A SINGLE OR MULTIPLE RADIOLOGIST(S)</b>
# 
# Summary to be done later...
# 
# **NOTE: Same caveat as plotting based on class. Only bounding boxes annotated by the specified radiologist will be plotted**<br>
# **NOTE: As radiologists often annotate `No tissue`, images may not contain ANY bounding boxes**
# 
# ---
# 
# More detail to come later

# In[69]:


train_data.plot_radiologists(rad_id_list=["R8"], verbose=False)


# <h3 style="text-align: font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;">5.2  VISUALIZE EACH ABNORMALITY </h3>
# 
# ---
# 
# We will leverage our created class to visualize 4 examples of every class...
# 
# 
# * Aortic Enlargement <i><sub>(CLASS-ID: 0)</sub></i>
# * Atelectasis <i><sub>(CLASS-ID: 1)</sub></i>
# * Calcification <i><sub>(CLASS-ID: 2)</sub></i>
# * Cardiomegaly <i><sub>(CLASS-ID: 3)</sub></i>
# * Consolidation <i><sub>(CLASS-ID: 4)</sub></i>
# * ILD <i><sub>(CLASS-ID: 5)</sub></i>
# * Infiltration <i><sub>(CLASS-ID: 6)</sub></i>
# * Lung Opacity <i><sub>(CLASS-ID: 7)</sub></i>
# * Nodule/Mass <i><sub>(CLASS-ID: 8)</sub></i>
# * Other Lesion <i><sub>(CLASS-ID: 9)</sub></i>
# * Pleural Effusion <i><sub>(CLASS-ID: 10)</sub></i>
# * Pleural Thickening <i><sub>(CLASS-ID: 11)</sub></i>
# * Pneumothorax <i><sub>(CLASS-ID: 12)</sub></i>
# * Pulmonary Fibrosis <i><sub>(CLASS-ID: 13)</sub></i>
# * No Tissue Present <i><sub>(CLASS-ID: 14)</sub></i>

# <b style="text-decoration: underline; font-family: Verdana;">AORTIC ENLARGMENT - (0)</b>
# 
# Summary to be done later...
# 
# ---
# 
# More detail to come later

# In[78]:


train_data.plot_classes(class_list=[0,], n=4, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">ATELECTASIS - (1)</b>
# 
# Summary to be done later...
# 
# ---
# 
# More detail to come later

# In[79]:


train_data.plot_classes(class_list=[1,], n=4, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">CALCIFICATION - (2)</b>
# 
# Summary to be done later...
# 
# ---
# 
# More detail to come later

# In[80]:


train_data.plot_classes(class_list=[2,], n=4, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">CARDIOMEGALY - (3)</b>
# 
# Summary to be done later...
# 
# ---
# 
# More detail to come later

# In[81]:


train_data.plot_classes(class_list=[3,], n=4, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">CONSOLIDATION - (4)</b>
# 
# Summary to be done later...
# 
# ---
# 
# More detail to come later

# In[82]:


train_data.plot_classes(class_list=[4,], n=4, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">ILD - (5)</b>
# 
# Summary to be done later...
# 
# ---
# 
# More detail to come later

# In[83]:


train_data.plot_classes(class_list=[5,], n=4, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">INFILTRATION - (6)</b>
# 
# Summary to be done later...
# 
# ---
# 
# More detail to come later

# In[84]:


train_data.plot_classes(class_list=[6,], n=4, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">LUNG OPACITY - (7)</b>
# 
# Summary to be done later...
# 
# ---
# 
# More detail to come later

# In[85]:


train_data.plot_classes(class_list=[7,], n=4, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">NODULE/MASS - (8)</b>
# 
# Summary to be done later...
# 
# ---
# 
# More detail to come later

# In[86]:


train_data.plot_classes(class_list=[8,], n=4, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">OTHER LESION - (9)</b>
# 
# Summary to be done later...
# 
# ---
# 
# More detail to come later

# In[87]:


train_data.plot_classes(class_list=[9,], n=4, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">PLEURAL EFFUSION - (10)</b>
# 
# Summary to be done later...
# 
# ---
# 
# More detail to come later

# In[88]:


train_data.plot_classes(class_list=[10,], n=4, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">PLEURAL THICKENING - (11)</b>
# 
# Summary to be done later...
# 
# ---
# 
# More detail to come later

# In[89]:


train_data.plot_classes(class_list=[11,], n=4, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">PNEUMOTHORAX - (12)</b>
# 
# Summary to be done later...
# 
# ---
# 
# More detail to come later

# In[90]:


train_data.plot_classes(class_list=[12,], n=4, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">PULMONARY FIBROSIS - (13)</b>
# 
# Summary to be done later...
# 
# ---
# 
# More detail to come later

# In[91]:


train_data.plot_classes(class_list=[13,], n=4, verbose=False)


# <b style="text-decoration: underline; font-family: Verdana;">NO TISSUE - (14)</b>
# 
# Summary to be done later...
# 
# ---
# 
# More detail to come later

# In[92]:


train_data.plot_classes(class_list=[14,], n=4, verbose=False)


# <a style="text-align: font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; color: navy; background-color: #ffffff;" id="combining_annotations">6&nbsp;&nbsp;COMBINING/MERGING OVERLAPPING ANNOTATIONS (WIP)</a>
# 
# EXPLANATION COMING SOON – NOT SURE ABOUT THE HANDELING OF NODULES AND OTHER SMALL BBOXES THAT GET ENGULFED BY LARGER SIMILAR ANNOTATIONS

# In[96]:


# First, load the original dimensions from the CSV file
original_dimensions = load_image_dimensions('/data/mhedas/common/challenge_dataset/img_size.csv')

# Now fix the IOU calculation code - it doesn't need to change, but let's review it to be sure
def calc_iou(bbox_1, bbox_2):
    # This function is working on the original coordinate space, so no changes needed
    # determine the coordinates of the intersection rectangle
    x_left = max(bbox_1[0], bbox_2[0])
    y_top = max(bbox_1[1], bbox_2[1])
    x_right = min(bbox_1[2], bbox_2[2])
    y_bottom = min(bbox_1[3], bbox_2[3])
    # Check if bboxes overlap at all (if not return 0)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        # compute the area of both AABBs
        bbox_1_area = (bbox_1[2] - bbox_1[0]) * (bbox_1[3] - bbox_1[1])
        bbox_2_area = (bbox_2[2] - bbox_2[0]) * (bbox_2[3] - bbox_2[1])
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bbox_1_area + bbox_2_area - intersection_area)
        return iou

def redux_bboxes(annots):
    def get_inner_box(bboxes):
        xmin = max([box[0] for box in bboxes])
        ymin = max([box[1] for box in bboxes])
        xmax = min([box[2] for box in bboxes])
        ymax = min([box[3] for box in bboxes])
        if (xmax<=xmin) or (ymax<=ymin):
            return None
        else:
            return [xmin, ymin, xmax, ymax]
    valid_list_indices = []
    new_bboxes = []
    new_class_ids = []
    new_rad_ids = []
    for i, (class_id, rad_id, bbox) in enumerate(zip(annots["class_id"], annots["rad_id"], annots["bbox"])):
        intersecting_boxes = [bbox,]
        other_bboxes = [x for j,x in enumerate(annots["bbox"]) if j!=i]
        other_classes = [x for j,x in enumerate(annots["class_id"]) if j!=i]
        for j, (other_class_id, other_bbox) in enumerate(zip(other_classes, other_bboxes)):
            if class_id==other_class_id:
                iou = calc_iou(bbox, other_bbox)
                if iou>0.:
                    intersecting_boxes.append(other_bbox)
        if len(intersecting_boxes)>1:
            inner_box = get_inner_box(intersecting_boxes)
            if inner_box and inner_box not in new_bboxes:
                new_bboxes.append(inner_box)
                new_class_ids.append(class_id)
                new_rad_ids.append(rad_id)
    annots["bbox"] = new_bboxes
    annots["rad_id"] = new_rad_ids
    annots["class_id"] = new_class_ids
    return annots

# Make GT Dataframe
gt_df = train_df[train_df.class_id!=14]

# Apply Manipulations and Merger Functions
gt_df["bbox"] = gt_df.loc[:, ["x_min","y_min","x_max","y_max"]].values.tolist()
gt_df.drop(columns=["x_min","y_min","x_max","y_max"], inplace=True)
gt_df = gt_df.groupby(["image_id"]).agg({k:list for k in gt_df.columns if k !="image_id"}).reset_index()
gt_df = gt_df.apply(redux_bboxes, axis=1)

# Recreate the Original Dataframe Style
gt_df = gt_df.apply(pd.Series.explode).reset_index(drop=True).dropna()
gt_df["x_min"] = gt_df["bbox"].apply(lambda x: x[0])
gt_df["y_min"] = gt_df["bbox"].apply(lambda x: x[1])
gt_df["x_max"] = gt_df["bbox"].apply(lambda x: x[2])
gt_df["y_max"] = gt_df["bbox"].apply(lambda x: x[3])
gt_df.drop(columns=["bbox"], inplace=True)

# Add back in NaN Rows As A Single Annotation
gt_df = pd.concat([
    gt_df, train_df.loc[train_df['class_id'] == 14].drop_duplicates(subset=["image_id"])
]).reset_index(drop=True)

# Create TrainData instances with original dimensions
train_data = TrainData(train_df, TRAIN_DIR, original_dimensions=original_dimensions)


# In[97]:


gt_data = TrainData(gt_df, TRAIN_DIR, original_dimensions=original_dimensions)

# Display sample images
IMAGE_ID_LIST = gt_df[gt_df.class_id!=14].groupby("image_id") \
                                         .count() \
                                         .sort_values(by="class_id", ascending=False) \
                                         .index[0:100:20]

for i, IMAGE_ID in enumerate(IMAGE_ID_LIST):
    train_data.get_annotated_image(IMAGE_ID, annots=None, plot=True, plot_size=(18,22), 
                                   plot_title=f"ORIGINAL – IMG #{i+1}")
    gt_data.get_annotated_image(IMAGE_ID, annots=None, plot=True, plot_size=(18,22), 
                                plot_title=f"REDUX VERSION – IMG #{i+1}")


# In[ ]:




