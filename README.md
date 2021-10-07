# Multiple Snapshot Attack On Steganographic Filesystems

This repository contains code supporting A Multiple Snapshot Attack on Steganographic Filesystems.
To our knowledge this is the first paper that successfully carried out this class of attack.
We carry this attack out on simulated data from Artifice, a deniable storage system, though it is applicable to most, if not all, steganographic file systems.

The attack broadly works by distinguishing between the pattern of disk writes in a normal file system and the pattern produced by steganographic systems.
Most steganographic systems randomly write to disk, but this leaves changes that are made uniformly, thus the likelihood that a single block will change is quite high.
In contrast normal filesystems make far fewer non-contiguous writes, leaving an avenue for attack.
Our classifier works simply by computing the probability of a single, non-contiguous write, (called a singleton or a 1-chain) and training a simple logistic regression threshold over these probabilities.

## Prerequisites
This code was written for Python 3.7.5.
It also makes use of several packages.
These packages are found in `requirements.txt` and can be easily installed with `pip3 install -r requirements.txt`.

## `multiple-snapshot-attack.ipynb`
The attack itself is contained in this jupyter notebook.
The notebook runs through constructing the data from a raw list of change records,
constructing features, and classifying those features as either containing or not containing a hidden volume.

## `src`
The raw data and the processed data is hosted at https://files.ssrc.us/data/disk-change-data.zip and is downloaded in the notebook with `download-data.sh`.
`artifice_utils.py` contains functions to calculate the size and probability of survival of Artifice instances, a deniable storage system.
`experiment_utils.py` contains functions to generate training data and features and then train and evaluate the classifier.
`uniform_writes.py` contains functions to theoretically and empirically calculate the probability of a $n$-chain given a disk of size $s$,
and produce list of chains from raw change records.

## `results-pub`
This directory contains the results reported in our paper.
The notebook can be used to display these results by setting `results-path` to `"results-pub"`.
In order to not overwrite these results with the results of a new experiment `num_trials` should be `0`.

## Citation
If you use this data or code, we would appreciate you citing the following paper:
Fill me in...

