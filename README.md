# league-of-nations-archives-digitization-challenge-starter-kit
![CrowdAI-Logo](https://github.com/crowdAI/crowdai/raw/master/app/assets/images/misc/crowdai-logo-smile.svg?sanitize=true)

This is a starter kit for the [League of Nations archives digitization challenge](https://www.crowdai.org/challenges/league-of-nations-archives-digitization-challenge) on 
[crowdAI](https://www.crowdai.org).

# Problem Statement
This challenge is an image classification problem, where in the training set you are given 4692 images belonging to either `english` or `french`, and then you are provided 14216 images in the test set, where you are supposed to predict the class the said image belongs to.

# Dataset
The datasets are available in the [Dataset section of the challenge page](https://www.crowdai.org/challenges/league-of-nations-archives-digitization-challenge/dataset_files), and on following the links, you will have two files : 

* `train.tar.gz`
* `test.tar.gz`

`train.tar.gz` expands into a folder containing two subfolders, of the form : 

```
.
└── train
    ├── en (contains *.jpg images)
    └── fr (contains *.jpg images)
```
The folders `en` and `fr` have `.jpg` images belonging to the respective class.
For the rest of this starter kit you are encourage to download both the files, and extract them and place them in the `data/` directory to make the directory structure look like : 
```
.
└── data
    ├── test_images  (contains *.jpg images)
    └── train 
        ├── en (contains *.jpg images)
        └── fr (contains *.jpg images)
```

# Prediction file format
The predictions should be a valid CSV file with 14216 rows (one for each of the images in the test set), and the following headers :
```
filename, prob_en, prob_fr
```
where :    
* `filename` : filename of a single test file
* `prob_en` : the confidence `[0,1]` that this image belongs to the class `english`
* `prob_fr` : the confidence `[0,1]` that this image belongs to the class `french`

The sum of of `prob_en` and `prob_fr` for a single row should be less than 1.

# Random prediction
The you can use the script below to generate a sample submission, which should be saved at `random_prediction.csv`.
```python
#!/usr/bin/env python

import numpy as np
import os
import glob

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


LINES = []
LINES.append("filename,prob_en,prob_fr")
for _file_path in glob.glob("data/test_images/*.jpg"):
    probs = softmax(np.random.rand(2))
    LINES.append("{},{},{}".format(
        os.path.basename(_file_path),
        probs[0],
        probs[1]
    ))

fp = open("random_prediction.csv", "w")
fp.write("\n".join(LINES))
fp.close()
```

# Submission

Then you can submit on crowdAI, by going to the challenge page and clicking on `Create Submission`: 
![create_submission](https://i.imgur.com/dqWSOcn.png)


and then upload the file by clicking on `Browse file` at the bottom of the screen:



![browse_file](https://i.imgur.com/QXQcLeS.png)


and then finally, your submission should either be accepted, or the error shown : 


![feedback](https://i.imgur.com/DmSExeK.png)


**Best of Luck**

# Author
Sharada Mohanty <sharada.mohanty@epfl.ch>
