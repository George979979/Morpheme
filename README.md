# Morpheme
Using basic model and Bert to handle with Morpheme problem

## Installation
```shell
# Make a fresh virtual environment
python3 -m venv env
source env/bin/activate
# Install requirements
pip install requirements.txt
```

## Datasets
We have referenced our data from the Morpheme Segmentation 2022 Shared Task contributed by Barsuren, Peters, and Nicolai. They achieved word and sentence morpheme segmentation for eight different languages.

## Models
The layers system in our model consists of two main components: the BasicNeuralTagger and the BertMorphemeLetterModel. These components are designed to tackle different tasks and exhibit distinct layer configurations.

## How to train models
```shell
# Train a basic model on English
python main.py -c config/basic.json -t data/eng.train -d data/eng.dev -s "@@"
```
### Train the bert-enhanced model
```shell
# Train a multilingual BERT model on English
python main.py -c config/bert_multi.json -t data/eng.train -d data/eng.dev -s "@@"
```

