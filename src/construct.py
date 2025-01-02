import argparse

from src.generate import Creator
from src import utils

QUERY_PROMPT = "\nDon't output the response for the above instruction. Instead, you need to predict the number of tokens in your response. Output one number only."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./ckpts/vicuna-7b")
    parser.add_argument("--data-path", type=str, default="data/sharegpt-train-10k.json")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    model = Creator(args.model)
    data = utils.jload(args.data_path)
    for i in range(len(data)):
        # user
        data[i]["conversations"][0]["value"] += QUERY_PROMPT
        len_to_predict = len(model.tokenizer(data[i]["conversations"][1]["value"]).input_ids)
        data[i]["conversations"][1]["from"] = "gpt"
        data[i]["conversations"][1]["value"] = "{}".format(len_to_predict)

    output_path = args.data_path.replace(".json", "-instruct.json")
    utils.jdump(data, output_path)
