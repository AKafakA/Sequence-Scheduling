from src import utils
import argparse
import os
import json
from transformers import AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-data-path", type=str, default="data/sharegpt-val-10k-predicted.json")
    parser.add_argument("--tokenizer", type=str, default="./ckpts/vicuna-7b")
    args = parser.parse_args()
    return args



def generate_lens_files(
        length_output_file,
        prompt_lens,
        response_lens, 
        predicted_lens):
    import csv
    assert length_output_file.endswith('.csv')
    with open(length_output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['num_prefill_tokens', 'num_decode_tokens', 'num_total_tokens', 'pd_ratio', 'num_predicted_decode_tokens'])
        for prompt_len, response_len, predicted_len in zip(prompt_lens, response_lens, predicted_lens):
            if prompt_len > 0:
                pd_ratio = (response_len * 1.0) / prompt_len
            else:
                pd_ratio = 1
            writer.writerow([prompt_len, response_len, prompt_len + response_len, pd_ratio, predicted_len])
    print(f"Dataset with real response lens saved to {length_output_file}")


if __name__ == "__main__":
    args = parse_args()
    with open(args.val_data_path) as f:
        val_data=json.load(f)
    tokenizer_model= AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompt_lens = []
    response_lens = []
    predicted_lens = []

    for data in val_data:
        prompt_lens.append(len(tokenizer_model.tokenize(data["conversations"][0]["value"])))
        response_lens.append(len(tokenizer_model.tokenize(data["conversations"][1]["value"])))
        predicted_lens.append(data['predicted_length'])

    output_file_name = args.val_data_path.replace(".json", "-len.csv")
    generate_lens_files(output_file_name, prompt_lens, response_lens, predicted_lens)
