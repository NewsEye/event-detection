## ACE2005 Event Detection

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

#### Data format:

```
DOC_ID EVENT_TYPE EVENT_SUBTYPE UNUSED_OLD_INDEX SENTENCE TRIGGER_WORD LIST[(TRIGGER_WORD, TRIGGER_SUBTYPE, POSITION_START,POSITION_END, 'EVENT_ID')]

AFP_ENG_20030401.0476	Personnel	Nominate	9	british Chancellor of the Exchequer Gordon Brown on Tuesday named the current head of the country 's energy regulator as the new chairman of finance watchdog the Financial Services Authority -LRB- FSA -RRB- .	named	[('named','Nominate','202,206','EV-1')]
AFP_ENG_20030401.0476	Personnel	Start-Position	5	former senior banker Callum McCarthy begins what is one of the most important jobs in London 's financial world in September , when incumbent Howard Davies steps down .	begins	[('begins','Start-Position','377,382','EV-2')]
AFP_ENG_20030401.0476	Personnel	End-Position	0	former senior banker Callum McCarthy begins what is one of the most important jobs in London 's financial world in September , when incumbent Howard Davies steps down .	Former	[('Former','End-Position','340,345','EV-3')]
AFP_ENG_20030401.0476	Personnel	End-Position	26	former senior banker Callum McCarthy begins what is one of the most important jobs in London 's financial world in September , when incumbent Howard Davies steps down .	steps down	[('steps down','End-Position','494,498','EV-4')]
AFP_ENG_20030401.0476	Personnel	End-Position	2	davies is leaving to become chairman of the London School of Economics , one of the best-known parts of the University of London .	leaving	[('leaving','End-Position','517,523','EV-5')]
AFP_ENG_20030401.0476	Personnel	Start-Position	4	davies is leaving to become chairman of the London School of Economics , one of the best-known parts of the University of London .	become	[('become','Start-Position','528,533','EV-6')]
```

The embeddings will be automatically downloaded, in the same manner as NLTK does.

```python
python extract_data.py --train data/train.txt --test data/test.txt --valid data/valid.txt --embeddings glove --output_directory data/processed/

--embeddings: [google, glove, fasttext, numberbatch]
```

### Run baseline model: Nugyen 2015

```python
cd ../
python train.py --directory data/ --train data/train.txt --test data/test.txt --valid data/valid.txt --embeddings glove  --generate_data

--embeddings: [google, glove, fasttext, numberbatch]
--models: [CNN2015, CNN2018, CNN2019, attention]

```

Weights saved in *weights/*
Results saved in *results/scores_baseline*





