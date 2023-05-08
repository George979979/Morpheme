# Morpheme
Using basic model and Bert to handle with Morpheme problem

## Installation
```shell
# Install requirements
pip install requirements.txt
```

## Datasets
We have referenced our data from the Morpheme Segmentation 2022 Shared Task contributed by Barsuren, Peters, and Nicolai. They achieved word and sentence morpheme segmentation for eight different languages. This is how the data look like:
| original words  | segmented words |
| ------------- | ------------- |
| jaspilites  | jaspilite@@s  |
| mouthed  | mouth@@ed  |
| deintercalated  | de@@intercalate@@eadd  |
| memorisers  | memoriser@@s |
| petereros  | peterero@@s |

## Models
The layers system in our model consists of two main components: the BasicNeuralTagger and the BertMorphemeLetterModel. These components are designed to tackle different tasks and exhibit distinct layer configurations.

## How to train models
### Train the basic model
```shell
# Train a basic model on English
python main.py -c config/basic.json -t data/eng.train.tsv -d data/eng.dev -s "@@"
# if you want to use cpu but not cuda
python main.py -c config/basic.json -t data/eng.train.tsv -d data/eng.dev -s "@@" -D cpu
```
### Train the bert-enhanced model
```shell
# Train a BERT model on English
python main.py -c config/eng/bert.json -t data/eng.train.tsv -d data/eng.dev -s "@@"
# if you want to use cpu but not cuda
python main.py -c config/eng/bert.json -t data/eng.train.tsv -d data/eng.dev -s "@@" -D cpu
```
## Output of the model
The output of basic model and Bert model have been stored in 'CNN_result' and 'BERT_result'
