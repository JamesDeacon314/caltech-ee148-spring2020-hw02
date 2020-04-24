import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    yA = max(box_1[0], box_2[0])
    xA = max(box_1[1], box_2[1])
    yB = min(box_1[2], box_2[2])
    xB = min(box_1[3], box_2[3])

    intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    area_1 = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
    area_2 = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)
    union = area_1 + area_2 - intersection

    iou = intersection / union

    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.)
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives.
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        pred_matches = np.zeros(len(pred))
        for i in range(len(gt)):
            matched_gt = False
            for j in range(len(pred)):
                if (pred[j][-1] > conf_thr):
                    # Since we are considering this contour we must FP
                    if (pred_matches[j] == 0): # Not already update
                        pred_matches[j] = -1
                    iou = compute_iou(pred[j][:4], gt[i])
                    if (iou > iou_thr):
                        TP += 1
                        pred_matches[j] = 1
                        matched_gt = True
                        break
            if not matched_gt:
                FN += 1

        for item in pred_matches:
            if (item == -1):
                FP += 1

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = 'data/hw02_preds'
gts_path = 'data/hw02_annotations'

# load splits:
split_path = 'data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data.
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)

with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:

    '''
    Load test data.
    '''

    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)

    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold.

plt.figure(1)
for iou in [0.25, 0.5, 0.75]:
    confidence_thrs = np.linspace(0,1,1000) # using (ascending) list of confidence scores as thresholds
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou, conf_thr=conf_thr)

    x = []
    y = []
    for i in range(len(confidence_thrs)):
        if (tp_train[i] + fp_train[i] != 0 and tp_train[i] + fn_train[i] != 0):
            y.append(tp_train[i] / (tp_train[i] + fp_train[i]))
            x.append(tp_train[i] / (tp_train[i] + fn_train[i]))
        else:
            x.append(0.0)
            y.append(0.0)

    plt.plot(x, y, '-.', label="IoU = " + str(iou))
    plt.legend()

plt.xlabel("Recall : TP/(TP + FN)")
plt.ylabel("Precision : TP/(TP + FP)")
plt.title("Training set PR curves")
plt.savefig("trainingPR.png")
plt.show()


# Plot training set PR curves

if done_tweaking:
    plt.figure(1)
    for iou in [0.25, 0.5, 0.75]:
        confidence_thrs = np.linspace(0,1,1000) # using (ascending) list of confidence scores as thresholds
        tp_test = np.zeros(len(confidence_thrs))
        fp_test = np.zeros(len(confidence_thrs))
        fn_test = np.zeros(len(confidence_thrs))
        for i, conf_thr in enumerate(confidence_thrs):
            tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=iou, conf_thr=conf_thr)

        x = []
        y = []
        for i in range(len(confidence_thrs)):
            if (tp_test[i] + fp_test[i] != 0 and tp_test[i] + fn_test[i] != 0):
                y.append(tp_test[i] / (tp_test[i] + fp_test[i]))
                x.append(tp_test[i] / (tp_test[i] + fn_test[i]))

        plt.plot(x, y, '-.', label="IoU = " + str(iou))
        plt.legend()

    plt.xlabel("Recall : TP/(TP + FN)")
    plt.ylabel("Precision : TP/(TP + FP)")
    plt.title("testing set PR curves")
    plt.savefig("testingPR.png")
    plt.show()
