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

The folders `class-X` have `.jpg` images belonging to the respective class. These class ids are mapped to there class names in `class_idx_mapping.csv`. 

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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


LINES = []
LINES.append('filename,agkistrodon_contortrix,agkistrodon_piscivorus,boa_imperator,carphophis_amoenus,charina_bottae,coluber_constrictor,crotalus_adamanteus,crotalus_atrox,crotalus_horridus,crotalus_pyrrhus,crotalus_ruber,crotalus_scutulatus,crotalus_viridis,diadophis_punctatus,haldea_striatula,heterodon_platirhinos,hierophis_viridiflavus,lampropeltis_californiae,lampropeltis_triangulum,lichanura_trivirgata,masticophis_flagellum,natrix_natrix,nerodia_erythrogaster,nerodia_fasciata,nerodia_rhombifer,nerodia_sipedon,opheodrys_aestivus,opheodrys_vernalis,pantherophis_alleghaniensis,pantherophis_emoryi,pantherophis_guttatus,pantherophis_obsoletus,pantherophis_spiloides,pantherophis_vulpinus,pituophis_catenifer,regina_septemvittata,rhinocheilus_lecontei,storeria_dekayi,storeria_occipitomaculata,thamnophis_elegans,thamnophis_marcianus,thamnophis_ordinoides,thamnophis_proximus,thamnophis_radix,thamnophis_sirtalis')

for _file_path in glob.glob("round1/*.jpg"):
    probs = softmax(np.random.rand(45))
    LINES.append(",".join([os.path.basename(_file_path)] + list(softmax(np.random.rand(45)))))

fp = open("random_prediction.csv", "w")
fp.write("\n".join(LINES))
fp.close()
```

**Best of Luck**

# Author
Sharada Mohanty <sharada.mohanty@epfl.ch>
