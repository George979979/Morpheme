from argparse import ArgumentParser
import json

import numpy as np

import torch
from transformers import BertTokenizer, BertModel


from read import read_infile
from dataset import BertSegmentDatasetReader
from dataloader import FieldBatchDataLoader
from network import BertMorphemeLetterModel
from training import do_epoch, initialize_metrics
from training import predict_with_model, decode_viterbi, measure_quality, normalize_target


argument_parser = ArgumentParser()
argument_parser.add_argument("-c", "--config-path", required=True)
argument_parser.add_argument("-D", "--device", default="cuda")
argument_parser.add_argument('-t', "--train-file", default=None)
argument_parser.add_argument("-d", "--dev-file", default=None)
argument_parser.add_argument("-T", "--test-file", default=None)
argument_parser.add_argument("-o", "--output-file", default=None)
argument_parser.add_argument("-l", "--language", default=None)
argument_parser.add_argument("-s", "--sep", default="/")
argument_parser.add_argument("-E", "--english", action="store_false")
argument_parser.add_argument("-R", "--no-reload", dest="reload", action="store_false")
argument_parser.add_argument("-m", "--models-number", default=1, type=int)
argument_parser.add_argument("-v", "--viterbi_decoding", action="store_true")



def read_config(config_path: str):
    DEFAULT_CONFIG = {
        "field": "bmes_labels", "d": 5, "max_morpheme_length": 7
    }
    DEFAULT_TRAIN_PARAMS = {"nepochs": 25, "checkpoint": "checkpoint.pt",
                            "combined_epochs": 0, "unimorph_task_weight": 0.1}
    with open(config_path, "r", encoding="utf8") as fin:
        config = json.load(fin)
    bert_model = config.pop("bert_model", None)
    model_type = config.pop("model_type", "bert")
    if bert_model or "embeddings" in config:
        if bert_model:
            config["vocab"] = BertTokenizer.from_pretrained(bert_model).vocab
        if model_type in ["random", "tokenization"]:
            with torch.no_grad():
                matrix = BertModel.from_pretrained(bert_model).embeddings.word_embeddings.weight.cpu().numpy()
            std = np.sqrt(((matrix - matrix.mean()) ** 2).mean())
            if "vocab_size" not in config:
                config["vocab_size"] = 768
            config["embeddings"] = np.random.normal(scale=std, size=(len(config["vocab"]), config["vocab_size"]))
        else:
            with torch.no_grad():
                config["embeddings"] = BertModel.from_pretrained(bert_model).embeddings.word_embeddings.weight.cpu().numpy()
            config["vocab_size"] = 768
        config["use_bert"] = True
    else:
        config["use_bert"] = False
        config["vocab"], config["embeddings"] = None, None
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value
    if "model_params" not in config:
        config["model_params"] = dict()
    config["model_params"]["use_bert"] = config["use_bert"]
    if config["use_bert"] and config["model_params"].get("use_subtoken_weights"):
        config["model_params"]["subtoken_vocab_size"] = len(config["embeddings"])
    if "train_params" not in config:
        config["train_params"] = dict()
    for key, value in DEFAULT_TRAIN_PARAMS.items():
        if key not in config["train_params"]:
            config["train_params"][key] = value
    return config

def append_model_number(s, index):
    if "." in s:
        stem, suffix = s.rsplit(".", maxsplit=1)
    else:
        stem, suffix = s, ""
    return stem + f"_{index}{suffix}"

if __name__ == "__main__":
    args = argument_parser.parse_args()
    # reading data
    train_data = read_infile(args.train_file, language=args.language, morph_sep=args.sep) if args.train_file is not None else None
    dev_data = read_infile(args.dev_file, language=args.language, morph_sep=args.sep) if args.dev_file is not None else None
    test_data = read_infile(args.test_file, language=args.language, morph_sep=args.sep) if args.test_file is not None else dev_data
    # reading model config
    config = read_config(args.config_path)
    bert_params = {key: config[key] for key in ["vocab", "embeddings"]}
    dataset_params = {key: config[key] for key in ["d", "max_morpheme_length", "field"]}
    # building datasets
    train_dataset = BertSegmentDatasetReader(train_data, **bert_params, **dataset_params)
    dev_dataset = BertSegmentDatasetReader(dev_data, **bert_params, **dataset_params,
                                           letters=train_dataset.letters_, labels=train_dataset.labels_)
    train_dataloader = FieldBatchDataLoader(train_dataset, device=args.device)
    dev_dataloader = FieldBatchDataLoader(dev_dataset, device=args.device)
    cls = BertMorphemeLetterModel
    cls_params = {
        "vocab_size": config.get("vocab_size"),
        "labels_number": len(train_dataset.labels_),
        "letters_number": len(train_dataset.letters_),
        "d": train_dataset.output_dim,
        "device": args.device,
        "task_weights": dict(),
    }

    qualities, viterbi_qualities = [], []
    for model_number in range(args.models_number):
        # building the model
        model = cls(**cls_params, **config["model_params"])
        # training the model
        best_val_acc = 0.0
        checkpoint = config["train_params"]["checkpoint"]
        if args.models_number > 1 and checkpoint is not None:
            checkpoint = append_model_number(checkpoint, model_number+1)
        metrics = initialize_metrics()
        for epoch in range(config["train_params"]["nepochs"]):
            if epoch >= config["train_params"]["combined_epochs"]:
                dataloader = train_dataloader
            do_epoch(model, dataloader, mode="train", epoch=epoch + 1, notebook=False)
            epoch_metrics = do_epoch(model, dev_dataloader, mode="validate", epoch=epoch + 1, notebook=False)
            if checkpoint is not None and epoch_metrics["accuracy"] > best_val_acc:
                best_val_acc = epoch_metrics["accuracy"]
                torch.save(model.state_dict(), checkpoint)
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint))
            do_epoch(model, dev_dataloader, mode="validate", epoch="evaluate", notebook=False)
        # evaluating on test
        test_dataset = BertSegmentDatasetReader(test_data, **bert_params, **dataset_params,
                                                letters=train_dataset.letters_, labels=train_dataset.labels_)
        corr_labels = ["".join(word_data[config["field"]]) for word_data in test_dataset.data]
        full_predictions = predict_with_model(model, test_dataset)
        raw_predictions = [elem["labels"] for elem in full_predictions]
        predictions = ["".join(x[0] for x in normalize_target(target)) for target in raw_predictions]
        viterbi_output = [decode_viterbi(elem["probs"], test_dataset.labels_) for elem in full_predictions]
        viterbi_predictions = ["".join([elem[0] for elem in states]) for states, _ in viterbi_output]
        corr_labels = ["".join(word_data[config["field"]]) for word_data in test_dataset.data]
        qualities.append(measure_quality(corr_labels, predictions, english_metrics=args.english, measure_last=False))
        viterbi_qualities.append(measure_quality(corr_labels, viterbi_predictions, english_metrics=args.english, measure_last=False))
        # if args.output_file is not None:
        #     output_file = append_model_number(args.output_file, model_number+1)
        #     words = [word_data["word"] for word_data in test_dataset.data]
        #     write_output(words, corr_labels, predictions, args.output_file)
    for quality in qualities:
        print(" ".join("{}:{:.2f}".format(elem[0], 100*elem[1]) for elem in quality.items()))
    print("")
    for quality in viterbi_qualities:
        print(" ".join("{}:{:.2f}".format(elem[0], 100*elem[1]) for elem in quality.items()))
    print("")
    if len(qualities) > 1:
        mean_quality, std_quality = dict(), dict()
        for key in qualities[0]:
            mean_quality[key] = np.mean([elem[key] for elem in qualities])
            std_quality[key] = np.std([elem[key] for elem in qualities])
        print(" ".join(
            "{}:{:.2f}({:.2f})".format(key, 100*value, 100*std_quality[key])  for key, value in mean_quality.items()
        ))
    if len(viterbi_qualities) > 1:
        mean_quality, std_quality = dict(), dict()
        for key in viterbi_qualities[0]:
            mean_quality[key] = np.mean([elem[key] for elem in viterbi_qualities])
            std_quality[key] = np.std([elem[key] for elem in viterbi_qualities])
        print(" ".join(
            "{}:{:.2f}({:.2f})".format(key, 100*value, 100 * std_quality[key]) for key, value in mean_quality.items()
        ))