![AIcrowd](https://s3.eu-central-1.amazonaws.com/aicrowd-static/misc/AIcrowd-flat.png)
# snake-species-identification-challenge-starter-kit

This is a starter kit for the [Snakes Species Identification Challenge](https://www.aicrowd.com/challenges/snake-species-identification-challenge) on 
[AIcrowd](https://www.aicrowd.com).

# Problem Statement
In this challenge you will be provided with a dataset of RGB images of snakes, and their corresponding species (class). The goal is to train a classification model.

The difficulty of the challenge relies on the dataset characteristics, as there might be a high intraclass variance for certain classes and a low interclass variance among others, as shown in the examples from the Datasets section. Also, the distribution of images between class is not equal for all classes: the class with the most images has 12201, while the class with the fewest images has 17. (top & bottom 15 classes shared below - based on training data)

| scientific_name             | Count | scientific_name              | Count |
|-----------------------------|-------|------------------------------|-------|
| thamnophis-sirtalis         | 12201 | agkistrodon-laticinctus      | 17    |
| storeria-dekayi             | 9708  | bitis-armata                 | 17    |
| pantherophis-obsoletus      | 8767  | bothrocophias-microphthalmus | 17    |
| crotalus-atrox              | 8679  | chironius-bicarinatus        | 17    |
| pituophis-catenifer         | 7113  | geophis-hoffmanni            | 17    |
| nerodia-sipedon             | 6436  | hebius-miyajimae             | 17    |
| agkistrodon-contortrix      | 4946  | lycodon-effraenis            | 17    |
| diadophis-punctatus         | 4406  | macrovipera-schweizeri       | 17    |
| lampropeltis-triangulum     | 4326  | naja-pallida                 | 17    |
| pantherophis-alleghaniensis | 4282  | philothamnus-punctatus       | 17    |
| nerodia-erythrogaster       | 4276  | sibynophis-collaris          | 17    |
| lampropeltis-californiae    | 4158  | spalerosophis-dolichospilus  | 17    |
| agkistrodon-piscivorus      | 4011  | thamnophis-chrysocephalus    | 17    |
| opheodrys-aestivus          | 3931  | agkistrodon-taylori          | 18    |
| crotalus-horridus           | 3571  | boiga-beddomei               | 18    |

The distribution of Snake Species and their image count looks as follows:

![Distribution of snake species](https://i.imgur.com/5QYBhKY.png)

To keep the barrier to entry much lower and demonstrate that an approach works well, we started with 85 species and 130150 images in previous round, and these numbers are increasing with every round. The idea would be then to renew the challenge every 4 months in order to get closer to our final goal, which is to build an algorithm which best predicts which antivenin should be given (if any) when given a specific image.

# Dataset
The datasets are available in the [Resources section of the challenge page](https://www.aicrowd.com/challenges/snake-species-identification-challenge/dataset_files), and on following the links, you have following files : 

* `train_images.tar.gz`
* `train_labels.tar.gz`
* `validate_images.tar.gz`
* `validate_labels.tar.gz`
* `validate_images_small.tar.gz`
* `validate_labels_small.tar.gz`

Where : 

* `train_images.tar.gz` untars into a folder containing `245185` images of snakes spread across `783` different snake species. 
* `train_labels.tar.gz` untars into a CSV with the following structure : 

| scientific_name           | country                  | hashed_id  |
|---------------------------|--------------------------|------------|
| crotalus-pyrrhus          | United States of America | f670636e2f |
| phyllorhynchus-decurtatus | United States of America | 5bfe5fa2ef |
| thamnophis-marcianus      | United States of America | 94d2da23c9 |
| boa-constrictor           | UNKNOWN                  | 871c3b709a |
| crotalus-atrox            | United States of America | e983981e77 |
| boa-imperator             | Mexico                   | ba9f7def25 |
| masticophis-flagellum     | United States of America | 2402a939c3 |
| coluber-constrictor       | United States of America | af5eacaac1 |
| storeria-dekayi           | United States of America | ba74f6f6c1 |

With the following columns : 

    - `hashed_id` : Unique ID of a single image (files are present in `train_images.tar.gz` with name `{hashed_id}.jpg`
    - `scientific_name` : Unique class name for the image in question
    - `country` : Country where the image was taken


* `validate_images.tar.gz` expands into a folder `.jpg` files representing a sample with similar distribution as test data. This has been provided to help you locally run validation phase over your model.
* `validate_labels.tar.gz` expands into a CSV file with the following structure : 

| scientific_name       | country                  | hashed_id  |
|-----------------------|--------------------------|------------|
| lycodon-ruhstrati     | UNKNOWN                  | bac3ff1139 |
| nerodia-sipedon       | Canada                   | 516a58ab5c |
| naja-nigricollis      | UNKNOWN                  | b8319014ed |
| crotalus-atrox        | United States of America | cfed281bac |
| symphimus-mayae       | Mexico                   | d237616cb2 |
| pantherophis-vulpinus | United States of America | b581db474b |
| storeria-dekayi       | United States of America | fad71aeca3 |
| opheodrys-aestivus    | United States of America | 3fd4dea662 |
| storeria-dekayi       | United States of America | 83402064a6 |


The files `validate_images_small.tar.gz` and `validate_labels_small.tar.gz` are **small subset** of validation data, just to try out your submission locally without doing heavy downloads.

Before moving into the next phase, it would be good to download the datasets from the above mentioned links, and organize them in the `./data` folder with the following folder structure : 

```
├── data
│   ├── validate_images_small
│   │   ├── 01978e1d8d.jpg
│   │   ├── 019d1e8cae.jpg
│   │   ├── 04a3809dda.jpg
│   │   ├── ..............
│   │   ├── ..............
│   │   ├── ..............
│   │   ├── fbb98a8213.jpg
│   │   ├── fc9fd55077.jpg
│   │   └── fce0ab02dd.jpg
│   └── validate_labels_small.csv
```
**NOTE** : The training related files and directories are excluded in the illustration above for simplicity.

# Prediction file format
The predictions should be a valid CSV file with the same number of rows as the number of images in the test set (listed also in the `test_metadata` file), and the header should be the `hashed_id` of each test case, and the probability distribution across all the valid snake species in this round. The `run.py` script has a list of the valid snake species for this round, which can also be created from the `scientific_name` column in the `train_labels.csv` file. Overall, the file is expected to have `86` columns (`1` for hashed_id and `85` for each of the included snake species). The sum of the probabilities across all the snake-species columns should be `< 1.0`.

# Random prediction
A sample script which generates a random prediction for the whole test set is included in the [run.py](run.py). The included inline comments better illustrate the structure expected. Please ensure to use the following environment variables : 

* `AICROWD_TEST_IMAGES_PATH`
* `AICROWD_TEST_METADATA_PATH`
* `AICROWD_PREDICTIONS_OUTPUT_PATH`

to get the path to the test images, the test metadata, and the final path where the prediction outputs are to be saved. 


# Submission

To submit to the challenge you'll need to ensure you've set up an appropriate repository structure, create a private git repository at https://gitlab.aicrowd.com with the contents of your submission, and push a git tag corresponding to the version of your repository you'd like to submit.

### Repository Structure

We have created this sample submission repository which you can use as reference.

#### aicrowd.json
Each repository should have a aicrowd.json file with the following fields:

```
{
    "challenge_id" : "snake-species-identification-challenge",
    "grader_id": "snake-species-identification-challenge",
    "authors" : ["aicrowd-user"],
    "description" : "Snakes Random Classification Agent",
    "debug" : "false"
}
```

This file is used to identify your submission as a part of the Snake Species Identification Challenge.  You must use the `challenge_id` and `grader_id` specified above in the submission. You can enable ["debug" mode](https://discourse.aicrowd.com/t/how-to-view-logs-for-your-submission/2370) for having quicker submission with 100 test image, for integration testing, those submissions score would not be counted toward leaderboard.

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
git clone https://github.com/AIcrowd/snake-species-identification-challenge-starter-kit snake-species-identification-challenge
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
* Sharada Mohanty (mohanty@aicrowd.com)
* Shivam Khandelwal (shivam@aicrowd.com)
