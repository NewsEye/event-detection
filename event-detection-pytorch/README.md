## ACE2005 Event Detection

The models available in this repository are:
* the baseline model: *Event detection and domain adaptation with convolutional neural networks.*, Nguyen, Thien Huu, and Ralph Grishman, 2015
* the baseline model with character embeddings___

Warning: the batch_size hyperparameter is very important! The best results are obtained with 128. Every other value will give lower results.
___


### Requirements

Preferably Anaconda, https://repo.continuum.io/, Python 3.x and an environment: 
```
conda create -n yourenvname python=3.6 anaconda
source activate yourenvname
```
If not, you your favorite library for creating python environments.
Once in your environement, execute: 

```
./install.sh
```
which installs the requirements and downloads models for SpaCy and data resources and models for NLTK.

### Pre-process data for trigger detection

The embeddings will be automatically downloaded, in the same manner as NLTK does.

```python
python extract_data.py --embeddings google

--embeddings: [google, glove, fasttext, numberbatch]
```

### Run baseline model: Nugyen 2015

```python
cd ../
python CNN_baseline.py --embeddings google

--embeddings: [google, glove, fasttext, numberbatch]
```

Weights saved in *weights/*
Results saved in *results/scores_baseline*




