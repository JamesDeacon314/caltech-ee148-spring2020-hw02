import os
import numpy as np
import json
from PIL import Image
import cv2
import math

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays)
    and returns a heatmap where each grid represents the output produced by
    convolution at each location. You can add optional parameters (e.g. stride,
    window_size, padding) to create additional functionality.
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    heatmap = np.zeros((n_rows, n_cols))
    for i in range(n_channels):
        for h in range(n_rows - np.shape(T)[0] + 1):
            for w in range(n_cols - np.shape(T)[1] + 1):
                subarray = I[h:h + np.shape(T)[0],w:w + np.shape(T)[1], i]
                heatmap[h, w] += np.sum(T[:, :, i] * subarray) / np.sum(np.square(T[:, :, i])) * 255

    heatmap = np.array(heatmap, dtype=np.uint8)
    return heatmap


def predict_boxes(maskr, image, heatmap):
    '''
    This function takes heatmap, an image, and a mask and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    # get contours
    contours, hierarchy = cv2.findContours(maskr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if contour is a circle
    for con in contours:
        perimeter = cv2.arcLength(con, True)
        area = cv2.contourArea(con)
        if 10 > area or area > 250:
            continue
        if perimeter == 0:
            break
        circularity = 4*math.pi*(area/(perimeter*perimeter))
        if 0.8 < circularity < 1.15:
            mask = np.zeros(maskr.shape, np.uint8)
            cv2.drawContours(mask, con, -1, 255, -1)
            if cv2.mean(image, mask=mask)[2] >= 90:
                mean_val = cv2.mean(image, mask=mask)
                if (mean_val[2] / (mean_val[1] + mean_val[0])) > 0.8:
                    bbox = cv2.boundingRect(con)
                    score = min(0.999, np.average(heatmap[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]) / 127)
                    output.append([bbox[1],bbox[0],bbox[1]+bbox[3],bbox[0]+bbox[2],score])

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>.
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>.
    The first four entries are four integers specifying a bounding box
    (the row and column index of the top left corner and the row and column
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    # Format the image
    image = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
    I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Hue thresholds
    min_sat = min(90, int(cv2.mean(hsv)[2]))
    lower_red1 = np.array([0, min_sat, min_sat])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, min_sat, min_sat])
    upper_red2 = np.array([180, 255, 255])
    lower_not_red = np.array([30, min_sat, min_sat])
    upper_not_red = np.array([150, 255, 255])

    # Mask generation
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskr = cv2.add(mask1, mask2)

    maskbg = cv2.bitwise_not(cv2.inRange(hsv, lower_not_red, upper_not_red))
    maskr = cv2.bitwise_and(maskr, maskbg)

    # Mask filtering
    kernele = np.ones((2,2),np.uint8)
    kernel = np.ones((1,1), np.uint8)
    kerneld = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7))
    maskr = cv2.erode(maskr,kernel,iterations=1)
    maskr = cv2.morphologyEx(maskr, cv2.MORPH_CLOSE, kerneld, iterations=1)
    maskr = cv2.dilate(cv2.erode(maskr,kernele,iterations=1),kernele,iterations=1)

    image = cv2.bitwise_and(image,image,mask = maskr)

    # read image using PIL:
    T = Image.open(os.path.join("mytemplate.jpg"))

    # convert to numpy array:
    T = np.asarray(T)
    T = cv2.cvtColor(T, cv2.COLOR_RGB2BGR)
    dim = (7, int(T.shape[0] / T.shape[1] * 7))
    T = cv2.resize(T, dim, interpolation = cv2.INTER_AREA)

    heatmap = compute_convolution(image, T)
    output = predict_boxes(maskr, I, heatmap)

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = 'data/RedLights2011_Medium'

# load splits:
split_path = 'data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = 'data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

    print(preds_train)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set.
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

        print(preds_test)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
