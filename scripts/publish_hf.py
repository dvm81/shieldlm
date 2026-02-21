#!/usr/bin/env python3
"""Upload ShieldLM dataset and/or model to HuggingFace Hub."""

import argparse

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


def publish_dataset(data_dir: str, repo_id: str):
    """Upload dataset splits to HuggingFace."""
    print(f"Loading dataset from {data_dir}...")
    train = Dataset.from_parquet(f"{data_dir}/train.parquet")
    val = Dataset.from_parquet(f"{data_dir}/val.parquet")
    test = Dataset.from_parquet(f"{data_dir}/test.parquet")

    ds = DatasetDict({"train": train, "validation": val, "test": test})

    print(f"Uploading to {repo_id}...")
    ds.push_to_hub(repo_id, private=False)
    print(f"Dataset published: https://huggingface.co/datasets/{repo_id}")


def publish_model(model_dir: str, repo_id: str):
    """Upload trained model to HuggingFace."""
    api = HfApi()
    print(f"Uploading model from {model_dir} to {repo_id}...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Model published: https://huggingface.co/models/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Publish ShieldLM artifacts to HuggingFace")
    sub = parser.add_subparsers(dest="command", required=True)

    ds_parser = sub.add_parser("dataset", help="Upload dataset")
    ds_parser.add_argument("--data-dir", default="./data/unified")
    ds_parser.add_argument("--repo-id", default="dmilush/shieldlm-prompt-injection")

    model_parser = sub.add_parser("model", help="Upload model")
    model_parser.add_argument("--model-dir", default="./models/deberta-v3-base-shieldlm")
    model_parser.add_argument("--repo-id", default="dmilush/shieldlm-deberta-base")

    args = parser.parse_args()

    if args.command == "dataset":
        publish_dataset(args.data_dir, args.repo_id)
    elif args.command == "model":
        publish_model(args.model_dir, args.repo_id)


if __name__ == "__main__":
    main()
