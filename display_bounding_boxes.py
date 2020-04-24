import os
import json
from PIL import Image
import numpy as np
import cv2
import sys

data_path = 'data/RedLights2011_Medium'
preds_path = 'data/hw02_preds'
anot_path = 'data/hw02_annotations'

with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    pred_boxes = json.load(f)

with open(os.path.join(anot_path,'annotations_train.json'),'r') as f:
    anot_boxes = json.load(f)


for img, box_set in pred_boxes.items():
    print(img)
    I = Image.open(os.path.join(data_path,img))
    I = np.asarray(I)

    image = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
    cv2.imshow("Initial Image", image)

    # Draw the boxes
    for i in range(len(box_set)):
        assert len(box_set[i]) == 5
        cv2.rectangle(image,(tuple([box_set[i][1],box_set[i][0]])),(tuple([box_set[i][3],box_set[i][2]])),(255, 255, 0),-1)

    box_set  = anot_boxes[img]
    box_set = np.array(box_set, dtype=np.uint32)
    for i in range(len(box_set)):
        assert len(box_set[i]) == 4
        cv2.rectangle(image,(tuple([box_set[i][1],box_set[i][0]])),(tuple([box_set[i][3],box_set[i][2]])),(255, 0, 255),2)

    cv2.imshow("Red Lights", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
