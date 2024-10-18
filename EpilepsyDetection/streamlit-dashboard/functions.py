

import pandas as pd
from radiomics import featureextractor
import cv2

import scipy.stats as stats


# This function gets as inputs an image and a mask, of type 'SimpleITK.SimpleITK.Image', and
# the list of radiomic features we are interested in, it returns a dataset with the features
# of the whole images.

def featurexImg(image, mask, ftype):
    df = pd.DataFrame()
    for feat in ftype:
        features_list = []
        # Define the feature extractor we're going to employ
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName(feat)

        # img = sitk.ReadImage(image_path, sitk.sitkFloat32)
        # msk = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        features = extractor.execute(image, mask, label=1)
        og_features = {key: value.item() for key, value in features.items() if key.startswith('original_')}
        features_list.append(og_features)

        # convert the list in a pd.DataFrame and returns it
        curr_df = pd.DataFrame(features_list)
        df = pd.concat([df, curr_df], axis=1)
    return df


def load_image(image, size):
  '''
  reads, rezises, converts to grayscale and normalizes the input path
  '''
  # image = cv2.imread(path)
  image = cv2.resize(image, (size,size))
  image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  image = image/255.
  return image


def find_boundaries(indices):

    if len(indices)==0:
        return []

    boundaries = []
    groups = []
    current_group = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_group.append(indices[i])
        else:
            groups.append(current_group)
            current_group = [indices[i]]

    # Append the last group
    groups.append(current_group)

    for i in range(0, len(groups) - 1, 2):
        boundaries.append(groups[i][0])  # First element of the current group
        boundaries.append(groups[i + 1][-1])  # Last element of the next group

    return boundaries


