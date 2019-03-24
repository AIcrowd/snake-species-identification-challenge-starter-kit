#!/usr/bin/env python

import numpy as np
import os
import glob

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


LINES = []

with open('data/class_idx_mapping.csv') as f:
	classes = ['filename']
	for line in f.readlines()[1:]:
		class_name = line.split(",")[0]
		classes.append(class_name) 

LINES.append(','.join(classes))

for _file_path in glob.glob("data/round1/*.jpg"):
	probs = softmax(np.random.rand(45))
	probs = list(map(str, probs))
	LINES.append(",".join([os.path.basename(_file_path)] + probs))

fp = open("random_prediction.csv", "w")
fp.write("\n".join(LINES))
fp.close()
