import csv 
import numpy as np

coco_categories = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
    10: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench', 
    15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 
    20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe', 
    26: 'backpack', 27: 'umbrella', 30: 'handbag', 31: 'tie', 32: 'suitcase', 
    33: 'frisbee', 34: 'skis', 35: 'snowboard', 36: 'sports ball', 37: 'kite', 
    38: 'baseball bat', 39: 'baseball glove', 40: 'skateboard', 41: 'surfboard', 
    42: 'tennis racket', 43: 'bottle', 45: 'wine glass', 46: 'cup', 47: 'fork', 
    48: 'knife', 49: 'spoon', 50: 'bowl', 51: 'banana', 52: 'apple', 
    53: 'sandwich', 54: 'orange', 55: 'broccoli', 56: 'carrot', 57: 'hot dog', 
    58: 'pizza', 59: 'donut', 60: 'cake', 61: 'chair', 62: 'couch', 
    63: 'potted plant', 64: 'bed', 66: 'dining table', 69: 'toilet', 
    71: 'tv', 72: 'laptop', 73: 'mouse', 74: 'remote', 75: 'keyboard', 
    76: 'cell phone', 77: 'microwave', 78: 'oven', 79: 'toaster', 
    80: 'sink', 81: 'refrigerator', 83: 'book', 84: 'clock', 85: 'vase', 
    86: 'scissors', 87: 'teddy bear', 88: 'hair drier', 89: 'toothbrush'
}

swapped_dict = {value: key for key, value in coco_categories.items()}
categories = []
with open("/home/gmc62/NSD/categories.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        categories.append([swapped_dict[item] for item in row if item])


matrix = np.zeros((73000, 90), dtype=int)

for i, category_ids in enumerate(categories):
    for j in category_ids:
        matrix[i, j] = 1

np.save("/home/gmc62/NSD/categories_one_hot.npy", matrix)
