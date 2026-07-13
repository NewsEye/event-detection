# Event Detection

Research code for multilingual event detection and information extraction, developed in the context of the [NewsEye project](https://www.newseye.eu/).

The repository brings together several complementary approaches:

- **DAniEL**, a multilingual string-based system for detecting epidemic-related events in press articles;
- **ACE2005 preprocessing** for converting ACE event annotations into sentence-level JSON, sequence-labeling, and question-answering formats;
- **neural event-trigger detection** with convolutional and attention-based PyTorch models;
- **Transformer-based sequence labeling** for named-entity and event-related extraction;
- **question-answering-based event extraction** using a SQuAD-style formulation; and
- **text-to-image conversion** for generating document images and studying OCR degradation.

> **Repository status**
>
> This is legacy research code. Most dependencies were pinned around 2019–2021 and are not expected to work unchanged with current versions of Python, PyTorch, TensorFlow, spaCy, or Transformers. Use a separate Python 3.6 or 3.7 environment for each component and consult the notes in [Known limitations](#known-limitations).

## Repository structure

```text
event-detection/
├── ace2005-preprocessing/       # ACE2005 parsing and format conversion
├── event-detection-daniel/      # Multilingual DAniEL epidemic-event detector
├── event-detection-entities/    # BERT/CamemBERT sequence-labeling models
├── event-detection-pytorch/     # CNN and attention event-trigger models
├── event-detection-qa/          # SQuAD-style QA event extraction
├── text2img/                    # Unicode text-to-document-image conversion
└── README.md
```

## Components

| Directory | Purpose | Main entry point |
|---|---|---|
| `event-detection-daniel/` | Language-independent epidemic surveillance from press articles | `daniel.py`, `process_corpus.py` |
| `ace2005-preprocessing/` | Parse ACE2005 annotations and generate model-ready data | `main.py`, `json_to_iob_and_qa.py` |
| `event-detection-pytorch/` | Train CNN and attention models for event-trigger detection | `extract_data.py`, `train.py` |
| `event-detection-entities/` | Train or evaluate BERT/CamemBERT-based sequence taggers | `main.py` |
| `event-detection-qa/` | Train extractive QA models on SQuAD-style event data | `run_squad.py` |
| `text2img/` | Convert Unicode text collections into document images | `text2img.py` |

## Installation

Clone the repository:

```bash
git clone https://github.com/NewsEye/event-detection.git
cd event-detection
```

Because the subprojects use different and sometimes incompatible dependency versions, **do not install all requirements into one environment**. Create one environment per component.

A typical legacy environment can be created with Conda:

```bash
conda create -n event-detection-py37 python=3.7
conda activate event-detection-py37
```

The exact dependencies and setup steps are described below.

---

## 1. DAniEL multilingual event detection

`event-detection-daniel/` contains a Python 3 port of [DAniEL](https://github.com/rundimeco/daniel), or **Data Analysis for Information Extraction in any Language**.

DAniEL was designed for epidemic surveillance in multilingual press collections. Instead of relying on language-specific parsers, it identifies repeated maximal substrings in salient article zones and matches them against disease and location resources. The approach is motivated by journalistic writing conventions, in which important information frequently appears near the beginning or end of an article.

### Requirements

- Python 3.7 recommended;
- [`justext`](https://pypi.org/project/jusText/) for optional HTML boilerplate removal.

```bash
cd event-detection-daniel
pip install justext
```

### Detect events in one document

```bash
python daniel.py \
  --language en \
  --document_path /path/to/article.txt \
  --verbose
```

Equivalent short options are available:

```bash
python daniel.py -l en -d /path/to/article.txt -v
```

Useful options include:

- `-l`, `--language`: two-letter language code;
- `-d`, `--document_path`: input document;
- `-r`, `--ratio`: relative substring threshold, default `0.8`;
- `-i`, `--isnot_clean`: apply HTML boilerplate removal;
- `-o`, `--out`: output file;
- `-v`, `--verbose`: print processing details;
- `-s`, `--showrelevant`: display information for relevant documents.

### Process a corpus

```bash
python process_corpus.py \
  --corpus /path/to/corpus.json \
  --language en \
  --ratio 0.8
```

To run evaluation immediately after corpus processing:

```bash
python process_corpus.py \
  --corpus /path/to/corpus.json \
  --language en \
  --ratio 0.8 \
  --evaluate
```

### Evaluate predictions

```bash
python evaluate.py \
  /path/to/ground_truth.json \
  /path/to/predictions.results
```

### Corpus annotation format

The corpus is represented as a JSON object indexed by document identifier. Each document should include a path and may include metadata and event annotations.

```json
{
  "document-001": {
    "document_path": "documents/article-001.txt",
    "source": "example-newspaper.org",
    "language": "en",
    "url": "https://example.org/article-001",
    "comment": "Optional note",
    "annotations": [
      ["dengue", "Jakarta"]
    ]
  }
}
```

Each annotation is a pair:

```text
[disease name, location]
```

An annotation example is available in `event-detection-daniel/docs/Indonesian_GL.json`. The source documents referenced by that file are not all distributed with this repository.

### Supported resources

The `resources/` directory contains disease and location lexicons for multiple languages, including Arabic, Bulgarian, Chinese, Dutch, English, French, German, Greek, Indonesian, Italian, Polish, Russian, Slovenian, Spanish, and Vietnamese.

---

## 2. ACE2005 preprocessing

`ace2005-preprocessing/` parses ACE2005 English annotations and converts them into formats used by the neural event-detection and QA components.

### Data availability

The ACE2005 corpus is **not included** in this repository. It must be obtained separately from the Linguistic Data Consortium and used according to its license.

### Expected input

`main.py` expects:

1. the ACE2005 English data directory;
2. a local `data_list.csv` file describing the train, development, and test split;
3. Stanford CoreNLP unpacked as `stanford-corenlp-full-2018-10-05/`;
4. the legacy spaCy English model available under the name `en`; and
5. an existing `output/` directory.

A split file should contain one ACE-relative file path per row:

```csv
split,path
train,nw/AFP_ENG_20030304.0250
dev,nw/AFP_ENG_20030316.0020
test,nw/AFP_ENG_20030401.0476
```


### Run preprocessing

```bash
cd ace2005-preprocessing
mkdir -p output
python main.py --data /path/to/ace_2005_td_v7/data/English
```

The script writes:

```text
output/train.json
output/dev.json
output/test.json
```

Each sentence-level record contains tokenized words, entity mentions, event triggers, arguments, and token spans.

### Generate sequence-labeling and QA data

`json_to_iob_and_qa.py` converts the JSON files into several derived files:

```text
<split>.tsv
<split>_semeval.json
<split>_semeval.txt
<split>_tags.txt
<split>_text.txt
```

Run it from the preprocessing directory:

```bash
python json_to_iob_and_qa.py
```

Before running it, update the hard-coded `FOLDER` variable in the script so that it points to your local ACE2005 `timex2norm/` directory.

---

## 3. PyTorch event-trigger detection

`event-detection-pytorch/` provides neural baselines for ACE2005 trigger detection. Available model choices are:

- `CNN2015`;
- `CNN2018`;
- `CNN2019`; and
- `attention`.

The implementation is associated with the work described in [Neural Methods for Event Extraction](https://theses.hal.science/tel-01943841/document).

### Environment

```bash
conda create -n event-pytorch python=3.6
conda activate event-pytorch
cd event-detection-pytorch
pip install -r requirements.txt
python -m nltk.downloader all
python -m spacy download en
```

The requirements pin legacy versions such as PyTorch 1.3.1, Keras 2.3.1, spaCy 2.2.3, and pandas 0.25.1.

### Input format

The model expects tab-separated event-trigger records with fields corresponding to:

```text
DOCUMENT_ID
EVENT_TYPE
EVENT_SUBTYPE
OLD_TRIGGER_INDEX
SENTENCE
TRIGGER_TEXT
TRIGGER_METADATA
```

Example:

```text
AFP_ENG_20030401.0476	Personnel	Nominate	9	British Chancellor Gordon Brown named the new chairman.	named	[('named','Nominate','202,206','EV-1')]
```

### Prepare embeddings and processed data

```bash
python extract_data.py \
  --train data/train.txt \
  --valid data/valid.txt \
  --test data/test.txt \
  --embeddings glove \
  --output_directory data/processed
```

Supported embedding identifiers in the preprocessing code include:

```text
google, glove, fasttext, numberbatch, bert
```

Large pretrained embeddings may be downloaded automatically and can require substantial disk space.

### Train a model

```bash
python train.py \
  --directory data \
  --processed_directory data/processed \
  --train data/train.txt \
  --valid data/valid.txt \
  --test data/test.txt \
  --embeddings glove \
  --model CNN2019 \
  --generate_data
```

Other model choices can be selected with:

```bash
--model CNN2015
--model CNN2018
--model attention
```

Common options include:

```text
--epochs 30
--batch_size 64
--lr 0.001
--gpu 0
--N 10
--max_len 57
--save_path results
```

The script creates the directory selected by `--save_path` (default: `results`). The legacy early-stopping helper can also write a `checkpoint.pt` file in the working directory.

---

## 4. Transformer sequence labeling

`event-detection-entities/` implements sequence-labeling models using BERT or CamemBERT embeddings with a CRF prediction layer. It supports:

- a direct `bert` model; and
- a `stacked` Transformer model with document-level contextual embeddings.

### Environment

```bash
conda create -n event-entities python=3.7
conda activate event-entities
cd event-detection-entities
pip install -r requirements.txt
```

The component uses legacy versions of PyTorch, FastNLP, Flair, spaCy, and Transformers.

### Dataset format

Input files are token-per-line TSV files with blank lines separating sentences. The reader is designed for HIPE-style columns such as:

```text
TOKEN	NE-COARSE-LIT	NE-COARSE-METO	NE-FINE-LIT	NE-FINE-METO	NE-FINE-COMP	NE-NESTED	NEL-LIT	NEL-METO	MISC
Wienstrasse	I-LOC	O	O	O	O	O	null	O	SpaceAfter
```

### Train

Create the model output directory first:

```bash
mkdir -p models/entity-model
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --directory models/entity-model \
  --pre_trained_model bert-base-cased \
  --train_dataset /path/to/train.tsv \
  --dev_dataset /path/to/dev.tsv \
  --test_dataset /path/to/test.tsv \
  --batch_size 4 \
  --language english \
  --model stacked \
  --num_layers 2 \
  --do_train
```

For French CamemBERT experiments:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --directory models/camembert-model \
  --pre_trained_model camembert-base \
  --train_dataset /path/to/train.tsv \
  --dev_dataset /path/to/dev.tsv \
  --test_dataset /path/to/test.tsv \
  --batch_size 4 \
  --language french \
  --model stacked \
  --num_layers 2 \
  --do_train
```

### Predict

```bash
python main.py \
  --directory models/entity-model \
  --pre_trained_model bert-base-cased \
  --train_dataset /path/to/train.tsv \
  --dev_dataset /path/to/dev.tsv \
  --test_dataset /path/to/test.tsv \
  --dataset_dir /path/to/unlabelled-tsv-files \
  --output_dir /path/to/predictions \
  --extension tsv \
  --saved_model models/entity-model/best/best_ \
  --batch_size 4 \
  --language english \
  --model stacked \
  --num_layers 2 \
  --do_eval
```

The prediction mode recursively processes files under `--dataset_dir` with the extension supplied through `--extension`, which defaults to `txt`.

---

## 5. Question-answering event extraction

`event-detection-qa/` formulates event extraction as extractive question answering. It is based on the legacy Hugging Face SQuAD training pipeline and uses SQuAD-style JSON generated by `ace2005-preprocessing/json_to_iob_and_qa.py`.

### Expected data

The generated data follows the SQuAD structure:

```json
{
  "data": [
    {
      "title": "ACE2005",
      "paragraphs": [
        {
          "context": "Sentence containing a possible event.",
          "qas": [
            {
              "id": "0",
              "question": "attack attacker target instrument time place",
              "is_impossible": false,
              "answers": [
                {
                  "text": "attacked",
                  "answer_start": 24
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

Negative examples use `"is_impossible": true` and an empty `answers` list.

### Train and evaluate

Install a dependency set compatible with the legacy `transformers` API used by the code. The entity component pins `transformers==2.11.0`, which is a suitable starting point for this directory as well.

```bash
mkdir -p outputs

python event-detection-qa/run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --output_dir outputs/event-qa \
  --data_dir ace2005-preprocessing/output \
  --train_file train_semeval.json \
  --predict_file dev_semeval.json \
  --do_train \
  --do_eval \
  --version_2_with_negative \
  --max_seq_length 384 \
  --doc_stride 128 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --overwrite_output_dir
```

The output directory contains model checkpoints and prediction files, including SQuAD-style prediction and n-best JSON files.

---

## 6. Unicode text-to-image conversion

`text2img/` converts text files into document images. It is based on [text2ImgDoc](https://github.com/nnkhoa/text2ImgDoc) and was modified to support Unicode text, including Chinese and Greek.

### Requirements

Install ImageMagick and ensure that the `convert` executable is available:

```bash
convert --version
```

### Run

Run the script from the `text2img/` directory so that its local font resource can be resolved:

```bash
cd text2img
python text2img.py /path/to/text-corpus
```

The script creates:

```text
output/notag/   # text with HTML tags removed
output/png/     # rendered PNG document images
```

HTML tag removal was written specifically for the DAniEL-style article data and may need adaptation for other markup.

---

## Recommended workflow

A typical ACE2005 experiment follows this sequence:

```text
ACE2005 source files
        │
        ▼
ace2005-preprocessing/main.py
        │
        ├── output/train.json
        ├── output/dev.json
        └── output/test.json
        │
        ▼
ace2005-preprocessing/json_to_iob_and_qa.py
        │
        ├── token-label TSV files ──► event-detection-entities/
        ├── trigger data ───────────► event-detection-pytorch/
        └── SQuAD-style JSON ───────► event-detection-qa/
```

The DAniEL and `text2img` components can also be used independently for multilingual epidemic-event detection and OCR-noise experiments.

## Known limitations

- The codebase is a collection of research prototypes rather than a unified Python package.
- Dependencies are old and conflict across components; separate environments are strongly recommended.
- `event-detection-daniel/` uses `time.clock()`, which was removed in Python 3.8. Use Python 3.7 or replace it with `time.perf_counter()`.
- `ace2005-preprocessing/main.py` contains an interactive `pdb.set_trace()` call in the event-processing loop. Remove or comment it out for unattended preprocessing.
- `ace2005-preprocessing/json_to_iob_and_qa.py` contains a hard-coded local ACE2005 path in the `FOLDER` variable. Replace it before use.
- `data_list.csv`, Stanford CoreNLP files, pretrained models, pretrained embeddings, and licensed ACE2005 data are not included.
- Some scripts assume that output or model directories already exist.
- GPU paths and legacy multi-GPU behavior have not been modernized.
- There is no automated test suite or continuous-integration configuration in the repository.
- No repository-level license file is included. Check the licenses of the original projects, datasets, and dependencies before redistributing or reusing the code.

## Reproducibility checklist

Before opening an issue, verify that:

- the correct component-specific environment is active;
- the expected data files exist at the supplied paths;
- output directories have been created;
- local hard-coded paths have been updated;
- the correct legacy pretrained model and tokenizer versions are installed;
- ACE2005 is available under a valid license; and
- CUDA, PyTorch, and GPU driver versions are mutually compatible.

## Related resources

- [NewsEye project](https://www.newseye.eu/)
- [Original DAniEL repository](https://github.com/rundimeco/daniel)
- [Neural Methods for Event Extraction](https://theses.hal.science/tel-01943841/document)
- [Original text2ImgDoc repository](https://github.com/nnkhoa/text2ImgDoc)

## Acknowledgements

This repository includes code developed or adapted in the context of NewsEye and builds on DAniEL, Hugging Face Transformers, FastNLP, PyTorch, spaCy, Stanford CoreNLP, and text2ImgDoc.
