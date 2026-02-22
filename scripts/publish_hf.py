#!/usr/bin/env python3
"""Upload ShieldLM dataset and/or model to HuggingFace Hub."""

import argparse

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


def publish_dataset(data_dir: str, repo_id: str, card_path: str = None):
    """Upload dataset splits and card to HuggingFace."""
    print(f"Loading dataset from {data_dir}...")
    train = Dataset.from_parquet(f"{data_dir}/train.parquet")
    val = Dataset.from_parquet(f"{data_dir}/val.parquet")
    test = Dataset.from_parquet(f"{data_dir}/test.parquet")

    ds = DatasetDict({"train": train, "validation": val, "test": test})

    print(f"Uploading to {repo_id}...")
    ds.push_to_hub(repo_id, private=False)

    # Upload dataset card (README.md) if provided
    if card_path:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=card_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Dataset card uploaded from {card_path}")

    print(f"Dataset published: https://huggingface.co/datasets/{repo_id}")


def publish_model(model_dir: str, repo_id: str, card_path: str = None):
    """Upload trained model to HuggingFace."""
    api = HfApi()
    print(f"Uploading model from {model_dir} to {repo_id}...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["checkpoint-*", "training_args.bin"],
    )

    if card_path:
        api.upload_file(
            path_or_fileobj=card_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Model card uploaded from {card_path}")

    print(f"Model published: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Publish ShieldLM artifacts to HuggingFace")
    sub = parser.add_subparsers(dest="command", required=True)

    ds_parser = sub.add_parser("dataset", help="Upload dataset")
    ds_parser.add_argument("--data-dir", default="./data/unified")
    ds_parser.add_argument("--repo-id", default="dmilush/shieldlm-prompt-injection")
    ds_parser.add_argument("--card", default="./hf_cards/dataset_card.md", help="Dataset card path")

    model_parser = sub.add_parser("model", help="Upload model")
    model_parser.add_argument("--model-dir", default="./models/deberta-v3-base-shieldlm")
    model_parser.add_argument("--repo-id", default="dmilush/shieldlm-deberta-base")
    model_parser.add_argument("--card", default=None, help="Model card path")

    args = parser.parse_args()

    if args.command == "dataset":
        publish_dataset(args.data_dir, args.repo_id, args.card)
    elif args.command == "model":
        publish_model(args.model_dir, args.repo_id, args.card)


if __name__ == "__main__":
    main()
