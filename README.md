# snake-species-identification-challenge-starter-kit
![AIcrowd-Logo](https://github.com/AIcrowd/AIcrowd/blob/master/app/assets/images/misc/aicrowd-horizontal.png)

This is a starter kit for the [Snakes Species Identification Challenge](https://www.aicrowd.com/challenges/snake-species-identification-challenge) on 
[AIcrowd](https://www.aicrowd.com).

# Problem Statement
In this challenge you will be provided with a dataset of RGB images of snakes, and their corresponding species (class). The goal is to train a classification model.

The difficulty of the challenge relies on the dataset characteristics, as there might be a high intraclass variance for certain classes and a low interclass variance among others, as shown in the examples from the Datasets section. Also, the distribution of images between class is not equal for all classes: the class with the most images has 11,092, while the class with the fewest images has 517.

For now, we would like to make the barrier to entry much lower and demonstrate that an approach works well on 45 species and 82,601 images. The idea would be then to renew the challenge every 4 months in order to get closer to our final goal, which is to build an algorithm which best predicts which antivenin should be given (if any) when given a specific image.

# Dataset
The datasets are available in the [Resources section of the challenge page](https://www.aicrowd.com/challenges/snake-species-identification-challenge/dataset_files), and on following the links, you will have 4 files : 

* `round1_test.tar.gz`
* `train.tar.gz`
* `sample_submission.csv`
* `class_idx_mapping.csv`


`train.tar.gz` expands into a folder containing 45 subfolders, of the form : 

```
.
└── train
    └── class-X
```

Here X is the class id which is a unique integer for every class. The folders `class-X` have `.jpg` images belonging to the respective class. 
The class labels in the training set and submission format are different. The class labels give for training are class ids which are integers whereas in the submission file the class name is expected with the probabilities.
To map the class ids to class names a file `class_idx_mapping.csv` is provided, where the column labels are `original_class` and `class_idx` which correspond to `class names` and `class ids` respectively. The label change can be made using this file.

`round1_test.tar.gz` expands into a folder round1 containing `.jpg` files to be predicted :

```
.
└── round1 (contains .jpg files)
```

# Prediction file format
The predictions should be a valid CSV file with 17731 rows (one for each of the images in the test set), and the following headers :
```
filename, agkistrodon_contortrix, agkistrodon_piscivorus, boa_imperator, carphophis_amoenus, charina_bottae, coluber_constrictor, crotalus_adamanteus, crotalus_atrox, crotalus_horridus, crotalus_pyrrhus, crotalus_ruber, crotalus_scutulatus, crotalus_viridis, diadophis_punctatus, haldea_striatula, heterodon_platirhinos, hierophis_viridiflavus, lampropeltis_californiae, lampropeltis_triangulum, lichanura_trivirgata, masticophis_flagellum, natrix_natrix, nerodia_erythrogaster, nerodia_fasciata, nerodia_rhombifer, nerodia_sipedon, opheodrys_aestivus, opheodrys_vernalis, pantherophis_alleghaniensis, pantherophis_emoryi, pantherophis_guttatus, pantherophis_obsoletus, pantherophis_spiloides, pantherophis_vulpinus, pituophis_catenifer, regina_septemvittata, rhinocheilus_lecontei, storeria_dekayi, storeria_occipitomaculata, thamnophis_elegans, thamnophis_marcianus, thamnophis_ordinoides, thamnophis_proximus, thamnophis_radix, thamnophis_sirtalis
```
where :
* `filename` : filename of a single test file
* `agkistrodon_contortrix` : the confidence `[0,1]` that this image belongs to the class `agkistrodon_contortrix`
* `agkistrodon_piscivorus` : the confidence `[0,1]` that this image belongs to the class `agkistrodon_piscivorus`

And so on for the rest of the classes

The sum of probabilities for a single row should be less than or equal to 1.

# Random prediction
The you can use the script below to generate a sample submission, which should be saved at `random_prediction.csv`.
```python
#!/usr/bin/env python

import numpy as np
import os
import glob

AICROWD_TEST_IMAGES_PATH = os.getenv('AICROWD_TEST_IMAGES_PATH', 'data/round1')
AICROWD_PREDICTIONS_OUTPUT_PATH = os.getenv('AICROWD_PREDICTIONS_OUTPUT_PATH', 'random_prediction.csv')

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

images_path = AICROWD_TEST_IMAGES_PATH + '/*.jpg'
for _file_path in glob.glob(images_path):
	probs = softmax(np.random.rand(45))
	probs = list(map(str, probs))
	LINES.append(",".join([os.path.basename(_file_path)] + probs))

fp = open(AICROWD_PREDICTIONS_OUTPUT_PATH, "w")
fp.write("\n".join(LINES))
fp.close()
```
# Starter code 

A jupyter notebook has been provided for the starter code of the snakes prediction challenge. This was based on an implementation of https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

# Round 2 Submission

To submit to the challenge you'll need to ensure you've set up an appropriate repository structure, create a private git repository at https://gitlab.aicrowd.com with the contents of your submission, and push a git tag corresponding to the version of your repository you'd like to submit.

### Repository Structure

We have created a sample submission repository which you can use as reference. You can find it [here](https://gitlab.aicrowd.com/aicrowd-bot/snakes_challenge_sample_submission)

#### aicrowd.json
Each repository should have a aicrowd.json file with the following fields:

```
{
    "challenge_id" : "snake-species-identification-challenge",
    "grader_id": "snake-species-identification-challenge",
    "authors" : ["aicrowd-user"],
    "description" : "Snakes Random Classification Agent"
}
```

This file is used to identify your submission as a part of the Snake Species Identification Challenge.  You must use the `challenge_id` and `grader_id` specified above in the submission. 

#### Submission environment configuration

You can specify your software environment by using all the [available configuration options of repo2docker](https://repo2docker.readthedocs.io/en/latest/config_files.html).

For example, to use Anaconda configuration files you can include an **environment.yml** file:
```
conda env export --no-build > environment.yml
```

It is important to include `--no-build` flag, which is important for allowing your Anaconda config to be replicable cross-platform.

#### Code Entrypoint

The evaluator will use `/home/aicrowd/run.sh` as the entrypoint. Please remember to have a `run.sh` at the root which can instantiate any necessary environment variables and execute your code. This repository includes a sample `run.sh` file.

### Submitting 
To make a submission, you will have to create a private repository on [https://gitlab.aicrowd.com](https://gitlab.aicrowd.com).

You will have to add your SSH Keys to your GitLab account by following the instructions [here](https://docs.gitlab.com/ee/gitlab-basics/create-your-ssh-keys.html).
If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

Then you can create a submission by making a *tag push* to your repository, adding the correct git remote and pushing to the remote:

```
cd snake-species-identification-challenge
# Add AICrowd git remote endpoint
git remote add aicrowd git@gitlab.aicrowd.com:<YOUR_AICROWD_USER_NAME>/snake-species-identification-challenge.git
git push aicrowd master

# Create a tag for your submission and push
git tag -am "submission-v0.1" submission-v0.1
git push aicrowd master
git push aicrowd submission-v0.1

# Note : If the contents of your repository (latest commit hash) does not change, 
# then pushing a new tag will not trigger a new evaluation.
```
You now should be able to see the details of your submission at : 
[gitlab.aicrowd.com/<YOUR_AICROWD_USER_NAME>/snake-species-identification-challenge/issues](gitlab.aicrowd.com/<YOUR_AICROWD_USER_NAME>/snake-species-identification-challenge/issues)

**Best of Luck**

# Author
Mridul Nagpal <mnagpal@aicrowd.com>
