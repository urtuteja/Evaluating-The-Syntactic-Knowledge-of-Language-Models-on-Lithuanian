#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import pandas as pd
import torch
import os
from huggingface_hub import login, whoami
from transformers import AutoTokenizer, AutoModelForCausalLM

def sentence_nll(sentence, tokenizer, model, device):
    """Compute negative log-likelihood (NLL) of a sentence."""
    enc = tokenizer(sentence, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids)

    #nll = outputs.loss.item() * (input_ids.size(1) - 1) # <-- TOTAL NLL
    nll = outputs.loss.item() # <-- MEAN NLL
    return nll

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--token", required=True, help="HuggingFace token")
    args = parser.parse_args()

    model_suffix = args.model.replace("/", "_")
    base, ext = os.path.splitext(args.output)
    output_file = f"{base}_{model_suffix}{ext}"
    summary_file = f"{base}_{model_suffix}_summary.txt"

    # Login
    login(token=args.token)
    print("Logged in as:", whoami()["name"])

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True,
                                              use_auth_token=True)
    
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 trust_remote_code=True,
                                                 use_auth_token=True,
                                                 device_map="auto",
                                                 offload_folder="offload",
                                                 torch_dtype=torch.float16)

    model.eval()

    if device.type == "mps":
        torch.mps.empty_cache()

    # Load data
    print(f"Loading input file: {args.input}")
    df = pd.read_csv(args.input, sep=";")

    # Column prefix based on model
    prefix = args.model.replace("/", "_")

    correct_nll_list, incorrect_nll_list, certainty_list, decision_list = [], [], [], []

    for index, row in df.iterrows():
        correct = str(row["correct_sentence"])
        incorrect = str(row["incorrect_sentence"])

        correct_nll = sentence_nll(correct, tokenizer, model, device)
        incorrect_nll = sentence_nll(incorrect, tokenizer, model, device)

        certainty = incorrect_nll - correct_nll
        decision = "correct" if correct_nll < incorrect_nll else "incorrect"

        correct_nll_list.append(correct_nll)
        incorrect_nll_list.append(incorrect_nll)
        certainty_list.append(certainty)
        decision_list.append(decision)

        print(f"Row {index}: decision={decision}, certainty={certainty:.3f}")

    # Add new columns
    df[f"{prefix}_correct_nll"] = correct_nll_list
    df[f"{prefix}_incorrect_nll"] = incorrect_nll_list
    df[f"{prefix}_certainty"] = certainty_list
    df[f"{prefix}_decision"] = decision_list

    # Save updated CSV
    df.to_csv(output_file, index=False, sep=";")
    print(f"\nSaved updated results to {output_file}\n")

    # Summaries
    df_all = df
    df_valid = df[df["note_exception"].fillna("").str.strip() == ""]

    def summarize(df_subset, name, fh):
        total = len(df_subset)
        correct_count = (df_subset[f"{prefix}_decision"] == "correct").sum()
        incorrect_count = (df_subset[f"{prefix}_decision"] == "incorrect").sum()
        accuracy = correct_count / total if total > 0 else 0.0

        lines = [f"\n{name}:",
                 f"Total sentences: {total}",
                 f"Correct decisions: {correct_count}",
                 f"Incorrect decisions: {incorrect_count}",
                 f"Accuracy: {accuracy:.3f}",
                 "\nAccuracy by error_name:",
                 df_subset.groupby("error_name")[f"{prefix}_decision"].apply(lambda x: (x == "correct").mean()).to_string(),
                 "\nAccuracy by linguistic_phenomenon:",
                 df_subset.groupby("linguistic_phenomenon")[f"{prefix}_decision"].apply(lambda x: (x == "correct").mean()).to_string(),
                 "\n" + "-" * 60]

        for line in lines:
            print(line)
            fh.write(line + "\n")

    with open(summary_file, "w", encoding="utf-8") as fh:
        summarize(df_all, "Including all rows", fh)
        summarize(df_valid, "Excluding exception rows", fh)

    print(f"\nSaved summary to {summary_file}\n")

if __name__ == "__main__":
    main()
