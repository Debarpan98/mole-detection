import cv2
import numpy as np
from copy import deepcopy
from scipy.interpolate import UnivariateSpline

def spreadLookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))

def warmImage(image):
    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 70, 150, 256])
    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 60, 110, 256])
    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    return cv2.merge((red_channel, green_channel, blue_channel))

def masknonskin_original(img_original,type='either',blur=1):
    img = deepcopy(img_original)
    img1 = deepcopy(img_original)


    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_RGBA  = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)


    if blur==1:
        imgblur = cv2.blur(img_original,(15,15))
        img_YCrCb_formask = cv2.cvtColor(imgblur, cv2.COLOR_BGR2YCrCb)
        img_RGBA_formask = cv2.cvtColor(imgblur, cv2.COLOR_BGR2RGBA)

    else:
        img_YCrCb_formask = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_RGBA_formask = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)


    # rgba mask
    R, G, B, A = cv2.split(img_RGBA_formask)
    RGBmask_rule = (R > 95) & (G > 40) & (B > 20) & (R > G) & (R > B) & (abs(R - G) > 15) & (A > 15)

    # ycrcb mask
    Y, Cr, Cb = cv2.split(img_YCrCb_formask)
    YCrCbmask_rule = (Cr > 135) & (Cb > 85) & (Y > 80) & (Cr <= (1.5862 * Cb) + 20) & (
                Cr >= (0.3448 * Cb) + 76.2069) & (Cr >= (-4.5652 * Cb) + 234.5652) & (Cr <= (-1.15 * Cb) + 301.75) & (
                                 Cr <= (-2.2857 * Cb) + 432.85)

    if type == "ycrcb":
        # only ycrcb mask
        img[~YCrCbmask_rule] = [0, 0, 0]
    elif type == "rgba":
        img_RGBA[~RGBmask_rule] = [0, 0, 0, 255]
        img = cv2.cvtColor(img_RGBA, cv2.COLOR_RGBA2BGR)

    elif type == "either": #if at least one filter detects skin, it's skin
        img_RGBA[~RGBmask_rule] = [0, 0, 0, 255]
        img = cv2.cvtColor(img_RGBA, cv2.COLOR_RGBA2BGR)
        img1[~YCrCbmask_rule] = [0, 0, 0]
        img=cv2.bitwise_or(img,img1)

    elif type == "both":  #if both filters detect skin, it's skin
        img_RGBA[~RGBmask_rule] = [0, 0, 0, 255]
        img = cv2.cvtColor(img_RGBA, cv2.COLOR_RGBA2BGR)
        img1[~YCrCbmask_rule] = [0, 0, 0]
        img=cv2.bitwise_and(img, img1)

    return img



def calculate_skin_percentage_general_new(img_original, type='either',blur=1):
    img=masknonskin_original(img_original, type, blur)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    totpixel = img.shape[0] * img.shape[1]
    blackpixel = totpixel - cv2.countNonZero(img)

    ratio = (totpixel - blackpixel) / totpixel

    return ratio

def masknonskin(img_original, type='either',blur=1):
    #ratio = calculate_skin_percentage_general_new(img_original)
    #if ratio >= 0.25:
    #    img_masked = masknonskin_original(img_original,type=type,blur=blur)
    #else:
    #    img_masked = masknonskin_original(cv2.cvtColor(warmImage(img_original), cv2.COLOR_RGB2BGR),type=type,blur=blur)
    img_masked = masknonskin_original(img_original, type=type, blur=blur)
    return img_masked


def masknonskin_updated(img_original,type='either',blur=1):
    img = deepcopy(img_original)

    #blurredimage = cv2.bilateralFilter(img, 70,150,150)
    #blurredimage = cv2.bilateralFilter(img, 40,75,75)


    if blur==1:
        imgblur = cv2.blur(img_original,(15,15))
        img_YCrCb_formask = cv2.cvtColor(imgblur, cv2.COLOR_BGR2YCrCb)
        img_RGBA_formask = cv2.cvtColor(imgblur, cv2.COLOR_BGR2RGBA)
        img_HSV = cv2.cvtColor(imgblur, cv2.COLOR_BGR2HSV)

    else:
        img_YCrCb_formask = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_RGBA_formask = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img_HSV=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    # rgba mask 1- Uniform daylight illumination
    R, G, B, A = cv2.split(img_RGBA_formask)
    RGBmask_rule = (R > 95) & (G > 40) & (B > 20) & (R > G) & (R > B) & (abs(R - G) > 15) & (A > 15)

    #rgba mask 2 - Flashlight or daylight lateral illumination
    RGBmask_rule_2 = (R > 220) & (G > 210) & (B > 170) & (G > B) & (R > B) & (abs(R - G) <= 15)

    # ycrcb mask
    Y, Cr, Cb = cv2.split(img_YCrCb_formask)
    #YCrCbmask_rule = (Cr <= 1.5862 * Cb + 20) & (Cr >= 0.3448 * Cb + 76.2069) & (Cr >= -1.005 * Cb + 234.5652) & (Cr <= -1.15 * Cb + 301.75) & (Cr <= -2.2857 * Cb + 432.85)

    H,S,V= cv2.split(img_HSV)
    HSV_mask_rule = (H<50) | (H>150)

    #ycrcb mask old
    YCrCbmask_rule = (Cr > 135) & (Cb > 85) & (Y > 80) & (Cr <= (1.5862 * Cb) + 20) & (Cr >= (0.3448 * Cb) + 76.2069) & (Cr >= (-4.5652 * Cb) + 234.5652) & (Cr <= (-1.15 * Cb) + 301.75) & (Cr <= (-2.2857 * Cb) + 432.85)

    #YCrCbmask_rule = (Cr > 135) & (Cb > 85) & (Y > 80) & (Cr <= (1.5862 * Cb) + 20) & (Cr >= (0.3448 * Cb) + 76.2069) & (Cr >= (-1.005 * Cb) + 234.5652) & (Cr <= (-1.15 * Cb) + 301.75) & (Cr <= (-2.2857 * Cb) + 432.85)


    if type == "ycrcb":
        # only ycrcb mask
        img = deepcopy(img_original)

        img[~YCrCbmask_rule] = [0, 0, 0]

    elif type == "rgba_1":
        img_copy = deepcopy(img_original)

        img_RGBA = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGBA)

        img_RGBA[~RGBmask_rule] = [0, 0, 0, 255]
        img = cv2.cvtColor(img_RGBA, cv2.COLOR_RGBA2BGR)

    elif type == "rgba_2":
        img_RGBA = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

        img_RGBA[~RGBmask_rule_2] = [0, 0, 0, 255]
        img = cv2.cvtColor(img_RGBA, cv2.COLOR_RGBA2BGR)

    elif type == "either": #if at least one filter detects skin, it's skin
        img_copy = deepcopy(img_original)
        img_copy_2 = deepcopy(img_original)
        img_copy_3 = deepcopy(img_original)

        img_RGBA_1 = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGBA)
        img_RGBA_2 = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGBA)

        img_RGBA_1[~RGBmask_rule] = [0, 0, 0, 255]
        img_1 = cv2.cvtColor(img_RGBA_1, cv2.COLOR_RGBA2BGR)

        img_RGBA_2[~RGBmask_rule_2] = [0, 0, 0, 255]
        img_2 = cv2.cvtColor(img_RGBA_2, cv2.COLOR_RGBA2BGR)

        img_copy_2[~YCrCbmask_rule] = [0, 0, 0]
        img_copy_3[~HSV_mask_rule] = [0, 0, 0]
        img=cv2.bitwise_and(cv2.bitwise_or(cv2.bitwise_or(img_1,img_2),img_copy_2),img_copy_3)

    elif type == "both":  #if both filters detect skin, it's skin
        img_RGBA = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

        img_RGBA[~RGBmask_rule] = [0, 0, 0, 255]
        img_1 = cv2.cvtColor(img_RGBA, cv2.COLOR_RGBA2BGR)

        img_copy = deepcopy(img_original)

        img_copy[~YCrCbmask_rule] = [0, 0, 0]
        img=cv2.bitwise_and(img_1, img_copy)


    return img