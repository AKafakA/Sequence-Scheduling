import argparse
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from transformers import AutoTokenizer
import pandas as pd
import logging
from src.generate import Creator
from src import utils
import random
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regression-model-name", type=str, default="roberta-base")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--train-data-path", type=str, default="data/sharegpt-train-10k.json")
    parser.add_argument("--val-data-path", type=str, default="data/sharegpt-val-10k.json")
    parser.add_argument("--output-dir", type=str, default="./ckpts/roberta-length-prediction")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-eval-examples", type=int, default=1000)
    args = parser.parse_args()
    return args

def generate_regression_dataframe(tokenizer_model, raw_data, num_sampled=-1):
    regression_dataset = []
    for i in range(len(raw_data)):
        new_data = []
        new_data.append(raw_data[i]["conversations"][0]["value"])
        len_to_predict = len(tokenizer_model.tokenize(raw_data[i]["conversations"][1]["value"]))
        new_data.append(len_to_predict)
        regression_dataset.append(new_data)
    if num_sampled > 0:
        regression_dataset = random.sample(regression_dataset, num_sampled)
    regression_df = pd.DataFrame(regression_dataset)
    regression_df.columns = ["text", "labels"]
    return regression_df, regression_dataset



if __name__ == "__main__":
    args = parse_args()
    train_data = utils.jload(args.train_data_path)
    val_data = utils.jload(args.val_data_path)
    tokenizer_model= AutoTokenizer.from_pretrained(args.tokenizer)

    train_data, _ = generate_regression_dataframe(tokenizer_model, train_data)
    num_eval_examples = args.num_eval_examples
    val_data, _ = generate_regression_dataframe(tokenizer_model, val_data, num_eval_examples)

    model_args = ClassificationArgs()
    model_args.num_train_epochs = args.epochs
    model_args.regression = True

    model = ClassificationModel(
    "roberta",
    args.regression_model_name,
    num_labels=1,
    args=model_args)

    model.train_model(train_data, output_dir=args.output_dir)
    result, model_outputs, wrong_predictions = model.eval_model(val_data)
    print(f"evaluation result: {result}")

    d_max = []
    assert len(val_data_list) == len(model_outputs)
    for i in range(len(val_data_list)):
        x = val_data_list[i][1]
        y = float(model_outputs(i))
        d_max.append(abs(x - y))

    diff = sum(d_max)
    acc_50t = sum([1 if x <= 50 else 0 for x in d_max])
    acc_100t = sum([1 if x <= 100 else 0 for x in d_max])

    print(f"# Samples: {num_eval_examples}")
    print(f"Error: {diff / num_eval_examples}")
    print(f"Acc-50: {acc_50t / num_eval_examples}")
    print(f"Acc-100: {acc_100t / num_eval_examples}")


    

