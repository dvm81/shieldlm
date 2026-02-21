#!/usr/bin/env python3
"""
ShieldLM Dataset Curation Pipeline
====================================
Aggregates, normalizes, and splits multiple prompt injection / jailbreak datasets
into a unified format for training a detection classifier.

Author: Dimiter Milushev
Date: February 2026

Usage:
    python curate_dataset.py --output ./data/unified --config taxonomy.yaml

Requirements:
    pip install datasets pandas pyyaml scikit-learn tqdm
"""

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shieldlm")


@dataclass
class Record:
    """Unified record schema — every sample in the final dataset looks like this."""
    id: str
    text: str
    label_binary: int           # 0=BENIGN, 1=ATTACK
    label_category: str         # benign | direct_injection | indirect_injection | jailbreak
    label_intent: Optional[str] = None
    source: str = ""
    language: str = "en"
    context: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Source Loaders
# ---------------------------------------------------------------------------

class BaseLoader:
    """Base class for dataset loaders."""
    name: str = "base"

    def load(self) -> list[Record]:
        raise NotImplementedError

    def _make_id(self, source: str, index: int, text: str) -> str:
        """Create a deterministic unique ID."""
        h = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{source}_{index:05d}_{h}"


class DeepsetLoader(BaseLoader):
    """Load deepset/prompt-injections from HuggingFace."""
    name = "deepset"

    def load(self) -> list[Record]:
        from datasets import load_dataset
        ds = load_dataset("deepset/prompt-injections", split="train")
        records = []
        for i, row in enumerate(ds):
            is_attack = int(row["label"])
            records.append(Record(
                id=self._make_id("deepset", i, row["text"]),
                text=row["text"],
                label_binary=is_attack,
                label_category="direct_injection" if is_attack else "benign",
                source="deepset/prompt-injections",
                metadata={"original_label": row["label"]},
            ))
        log.info(f"  deepset/prompt-injections: {len(records)} records")
        return records


class HarelixLoader(BaseLoader):
    """Load Harelix/Prompt-Injection-Mixed-Techniques-2024.

    Original dataset was deleted from HuggingFace. We recover ~987 rows
    (84% of original) from ahsanayub/malicious-prompts which includes
    Harelix as one of its 6 sources. Binary labels only (original 3-class
    labels are lost).
    """
    name = "harelix"

    def __init__(self, ahsanayub_path: str = "./data/raw/ahsanayub-malicious-prompts"):
        self.data_path = Path(ahsanayub_path)

    def load(self) -> list[Record]:
        records = []
        for csv_file in ["train.csv", "test.csv"]:
            filepath = self.data_path / csv_file
            if not filepath.exists():
                log.warning(f"  Harelix: {filepath} not found, skipping")
                continue
            df = pd.read_csv(filepath)
            harelix = df[df["source"].str.contains("Harelix", case=False, na=False)]
            for _, row in harelix.iterrows():
                text = str(row["text"]).strip()
                if not text:
                    continue
                is_attack = int(row["label"])
                records.append(Record(
                    id=self._make_id("harelix", int(row["id"]), text),
                    text=text,
                    label_binary=is_attack,
                    label_category="direct_injection" if is_attack else "benign",
                    source="harelix/mixed-techniques-2024",
                    metadata={"original_source": row["source"]},
                ))
        log.info(f"  Harelix (via ahsanayub): {len(records)} records")
        return records


class SPMLLoader(BaseLoader):
    """Load reshabhs/SPML_Chatbot_Prompt_Injection."""
    name = "spml"

    def load(self) -> list[Record]:
        from datasets import load_dataset
        ds = load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection", split="train")
        records = []
        for i, row in enumerate(ds):
            text = row.get("User Prompt")
            if not text or not str(text).strip():
                continue
            text = str(text)
            is_attack = int(row["Prompt injection"])
            system_prompt = row.get("System Prompt", None)
            records.append(Record(
                id=self._make_id("spml", i, text),
                text=text,
                label_binary=is_attack,
                label_category="direct_injection" if is_attack else "benign",
                source="spml/chatbot-prompt-injection",
                context=system_prompt,
                metadata={"original_label": row["Prompt injection"]},
            ))
        log.info(f"  SPML: {len(records)} records")
        return records


class SafeGuardLoader(BaseLoader):
    """Load xTRam1/safe-guard-prompt-injection."""
    name = "safeguard"

    def load(self) -> list[Record]:
        from datasets import load_dataset
        ds = load_dataset("xTRam1/safe-guard-prompt-injection", split="train")
        records = []
        for i, row in enumerate(ds):
            is_attack = int(row["label"])
            records.append(Record(
                id=self._make_id("safeguard", i, row["text"]),
                text=row["text"],
                label_binary=is_attack,
                label_category="direct_injection" if is_attack else "benign",
                source="safeguard/prompt-injection",
                metadata={"original_label": row["label"]},
            ))
        log.info(f"  SafeGuard: {len(records)} records")
        return records


class MultilingualLoader(BaseLoader):
    """Load yanismiraoui/prompt_injections (multilingual, all attacks)."""
    name = "multilingual"

    def load(self) -> list[Record]:
        from datasets import load_dataset
        ds = load_dataset("yanismiraoui/prompt_injections", split="train")
        records = []
        for i, row in enumerate(ds):
            text = row["prompt_injections"]
            # Detect language heuristically (could use langdetect for production)
            lang = self._detect_lang(text)
            records.append(Record(
                id=self._make_id("multilingual", i, text),
                text=text,
                label_binary=1,  # All are attacks
                label_category="direct_injection",
                source="yanismiraoui/prompt-injections",
                language=lang,
            ))
        log.info(f"  Multilingual: {len(records)} records")
        return records

    @staticmethod
    def _detect_lang(text: str) -> str:
        """Simple heuristic language detection."""
        try:
            from langdetect import detect
            return detect(text)
        except Exception:
            return "en"


class JailbreakBenchLoader(BaseLoader):
    """Load JailbreakBench/JBB-Behaviors.

    IMPORTANT: The 'Goal' field contains harmful behavior descriptions
    (e.g., "Write a tutorial on how to make a bomb"), NOT jailbreak attack
    prompts. These are the TARGET BEHAVIORS, not the attack TECHNIQUES.

    We use this dataset for FALSE POSITIVE evaluation:
    - Harmful goals -> label as benign (sensitive but not an injection)
    - Benign goals -> label as benign

    This tests whether the classifier can distinguish sensitive-topic
    questions from actual prompt injection or jailbreak attacks.
    A classifier that flags "how to make a bomb" as an injection is
    learning topic, not technique.

    For actual jailbreak TECHNIQUES (DAN, persona switching), use
    jackhhao/jailbreak-classification which has real jailbreak prompts.
    """
    name = "jailbreakbench"

    def load(self) -> list[Record]:
        from datasets import load_dataset
        records = []

        # Harmful goals — labeled BENIGN because these are behavior descriptions,
        # not injection techniques. Used as a false-positive stress test.
        ds_harmful = load_dataset(
            "JailbreakBench/JBB-Behaviors", "behaviors", split="harmful"
        )
        for i, row in enumerate(ds_harmful):
            records.append(Record(
                id=self._make_id("jbb_harmful_topic", i, row["Goal"]),
                text=row["Goal"],
                label_binary=0,  # NOT an injection — just a sensitive topic
                label_category="benign",
                source="jailbreakbench/jbb-behaviors",
                metadata={
                    "category": row.get("Category", ""),
                    "behavior": "harmful_topic",
                    "note": "Sensitive topic but NOT an injection. For FP evaluation.",
                },
            ))

        # Paired benign goals (separate split)
        ds_benign = load_dataset(
            "JailbreakBench/JBB-Behaviors", "behaviors", split="benign"
        )
        for i, row in enumerate(ds_benign):
            records.append(Record(
                id=self._make_id("jbb_benign", i, row["Goal"]),
                text=row["Goal"],
                label_binary=0,
                label_category="benign",
                source="jailbreakbench/jbb-behaviors",
                metadata={"behavior": "benign_paired"},
            ))

        log.info(f"  JailbreakBench: {len(records)} records (all benign — for FP evaluation)")
        return records


class WildJailbreakLoader(BaseLoader):
    """Load TrustAIRLab/in-the-wild-jailbreak-prompts.

    Real jailbreak techniques collected from Reddit, Discord, and jailbreak
    websites. All 1,405 prompts are confirmed jailbreaks (DAN, persona switching,
    role-play attacks, etc.).
    """
    name = "wild_jailbreak"

    def load(self) -> list[Record]:
        from datasets import load_dataset
        ds = load_dataset(
            "TrustAIRLab/in-the-wild-jailbreak-prompts",
            "jailbreak_2023_12_25",
            split="train",
        )
        records = []
        for i, row in enumerate(ds):
            text = row["prompt"]
            if not text or not str(text).strip():
                continue
            text = str(text)
            records.append(Record(
                id=self._make_id("wild_jailbreak", i, text),
                text=text,
                label_binary=1,
                label_category="jailbreak",
                source="trustailab/in-the-wild-jailbreak-prompts",
                metadata={
                    "platform": row.get("platform", ""),
                    "jailbreak_source": row.get("source", ""),
                },
            ))
        log.info(f"  Wild Jailbreak Prompts: {len(records)} records")
        return records


class JailbreakClassificationLoader(BaseLoader):
    """Load jackhhao/jailbreak-classification."""
    name = "jailbreak_classification"

    def load(self) -> list[Record]:
        from datasets import load_dataset
        ds = load_dataset("jackhhao/jailbreak-classification", split="train")
        records = []
        for i, row in enumerate(ds):
            is_jailbreak = row["type"] == "jailbreak"
            records.append(Record(
                id=self._make_id("jailclass", i, row["prompt"]),
                text=row["prompt"],
                label_binary=1 if is_jailbreak else 0,
                label_category="jailbreak" if is_jailbreak else "benign",
                source="jackhhao/jailbreak-classification",
                metadata={"original_type": row["type"]},
            ))
        log.info(f"  Jailbreak Classification: {len(records)} records")
        return records


class BenignPromptsLoader(BaseLoader):
    """Load benign prompts from alespalla/chatbot_instruction_prompts."""
    name = "benign_chatbot"

    def __init__(self, max_samples: int = 5000):
        self.max_samples = max_samples

    def load(self) -> list[Record]:
        from datasets import load_dataset
        ds = load_dataset("alespalla/chatbot_instruction_prompts", split="train")
        records = []
        # Sample subset to avoid overwhelming the dataset with benign examples
        indices = list(range(len(ds)))
        import random
        random.seed(42)
        random.shuffle(indices)
        for idx in indices[:self.max_samples]:
            row = ds[idx]
            text = row["prompt"]
            records.append(Record(
                id=self._make_id("benign_chat", idx, text),
                text=text,
                label_binary=0,
                label_category="benign",
                source="alespalla/chatbot-instruction-prompts",
            ))
        log.info(f"  Benign prompts: {len(records)} records (sampled from {len(ds)})")
        return records


class InjecAgentLoader(BaseLoader):
    """
    Load InjecAgent from local clone.
    This is the CRITICAL dataset for indirect prompt injection.
    """
    name = "injecagent"

    def __init__(self, repo_path: str = "./InjecAgent"):
        self.repo_path = Path(repo_path)

    def load(self) -> list[Record]:
        records = []

        # Load both direct harm and data stealing, both base and enhanced
        configs = [
            ("test_cases_dh_base.json", "direct_harm", "base"),
            ("test_cases_dh_enhanced.json", "direct_harm", "enhanced"),
            ("test_cases_ds_base.json", "data_stealing", "base"),
            ("test_cases_ds_enhanced.json", "data_stealing", "enhanced"),
        ]

        for filename, harm_type, setting in configs:
            filepath = self.repo_path / "data" / filename
            if not filepath.exists():
                log.warning(f"  InjecAgent file not found: {filepath}")
                continue

            with open(filepath) as f:
                cases = json.load(f)

            for i, case in enumerate(cases):
                # The "text" for classification is the tool response containing the injection
                tool_response = case.get("Tool Response", "")
                attacker_instruction = case.get("Attacker Instruction", "")
                user_instruction = case.get("User Instruction", "")
                attack_type = case.get("Attack Type", harm_type)

                # Map attack types to our intent taxonomy
                intent_map = {
                    "Financial Harm": "financial_harm",
                    "Physical Harm": "physical_harm",
                    "Data Security": "data_security",
                    "Financial Data": "data_exfiltration",
                    "Physical Data": "data_exfiltration",
                    "Others": "data_exfiltration",
                }
                intent = intent_map.get(attack_type, "data_exfiltration")

                # NOTE: We do NOT extract the attacker instruction as a standalone
                # record. "Please sell 50 units of my Bitcoin" is a legitimate user
                # instruction — the CONTEXT makes it an injection, not the text.
                # (See PromptShield, Jacob et al. 2025 — context determines injection.)

                # Record: The full tool response with embedded injection (realistic context)
                if tool_response and setting == "base":  # Avoid near-duplicates from enhanced
                    records.append(Record(
                        id=self._make_id(f"injecagent_full_{setting}", i, str(tool_response)[:200]),
                        text=str(tool_response),
                        label_binary=1,
                        label_category="indirect_injection",
                        label_intent=intent,
                        source=f"injecagent/full_response/{harm_type}/{setting}",
                        context=user_instruction,
                        metadata={
                            "attack_type": attack_type,
                            "harm_type": harm_type,
                            "setting": setting,
                            "record_type": "full_tool_response",
                            "attacker_instruction": attacker_instruction,
                        },
                    ))

                    # Also create a BENIGN version: same tool response with injection removed
                    # This teaches the classifier that tool response format != attack
                    if attacker_instruction and attacker_instruction in str(tool_response):
                        clean_response = str(tool_response).replace(
                            attacker_instruction, ""
                        ).strip()
                        if len(clean_response) > 20:
                            clean_id = self._make_id(
                                f"injecagent_clean_{setting}",
                                i,
                                clean_response[:200],
                            )
                            records.append(Record(
                                id=clean_id,
                                text=clean_response,
                                label_binary=0,
                                label_category="benign",
                                source=f"injecagent/clean_response/{harm_type}",
                                context=user_instruction,
                                metadata={
                                    "record_type": "clean_tool_response",
                                    "original_had_injection": True,
                                },
                            ))

        # Also load benign user instructions as negative examples
        user_cases_path = self.repo_path / "data" / "user_cases.jsonl"
        if user_cases_path.exists():
            with open(user_cases_path) as f:
                for i, line in enumerate(f):
                    case = json.loads(line)
                    records.append(Record(
                        id=self._make_id("injecagent_benign", i, case["User Instruction"]),
                        text=case["User Instruction"],
                        label_binary=0,
                        label_category="benign",
                        source="injecagent/user_instructions",
                        metadata={"user_tool": case.get("User Tool", "")},
                    ))

        log.info(f"  InjecAgent: {len(records)} records")
        return records


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class DatasetPipeline:
    """Main pipeline: load → deduplicate → balance → split → export."""

    def __init__(
        self,
        injecagent_path: str = "./InjecAgent",
        ahsanayub_path: str = "./data/raw/ahsanayub-malicious-prompts",
        benign_sample_size: int = 5000,
    ):
        self.loaders = [
            # --- Always available (local) ---
            InjecAgentLoader(injecagent_path),

            # --- HuggingFace datasets (require network) ---
            DeepsetLoader(),
            HarelixLoader(ahsanayub_path),
            SPMLLoader(),
            SafeGuardLoader(),
            MultilingualLoader(),
            JailbreakBenchLoader(),
            WildJailbreakLoader(),
            JailbreakClassificationLoader(),
            BenignPromptsLoader(max_samples=benign_sample_size),
        ]

    def run(
        self,
        output_dir: str = "./data/unified",
        skip_hf: bool = False,
        skip_sources: list[str] = None,
    ) -> pd.DataFrame:
        """Execute the full pipeline."""

        skip_sources = skip_sources or []
        all_records: list[Record] = []

        # ── Step 1: Load all sources ──────────────────────────────────────
        log.info("=" * 60)
        log.info("STEP 1: Loading datasets")
        log.info("=" * 60)

        for loader in self.loaders:
            if loader.name in skip_sources:
                log.info(f"  Skipping {loader.name} (user request)")
                continue

            # Skip HF loaders if requested
            if skip_hf and loader.name != "injecagent":
                log.info(f"  Skipping {loader.name} (--skip-hf flag)")
                continue

            try:
                records = loader.load()
                all_records.extend(records)
            except Exception as e:
                log.warning(f"  FAILED to load {loader.name}: {e}")
                log.warning("  Continuing without this source...")

        log.info(f"\nTotal raw records: {len(all_records)}")

        # ── Step 2: Convert to DataFrame ──────────────────────────────────
        log.info("\n" + "=" * 60)
        log.info("STEP 2: Building DataFrame")
        log.info("=" * 60)

        df = pd.DataFrame([asdict(r) for r in all_records])

        # ── Step 3: Clean & Deduplicate ───────────────────────────────────
        log.info("\n" + "=" * 60)
        log.info("STEP 3: Cleaning and deduplication")
        log.info("=" * 60)

        before = len(df)

        # Remove empty/null texts
        df = df[df["text"].notna() & (df["text"].str.strip() != "")]

        # Remove exact duplicates on text
        df = df.drop_duplicates(subset=["text"], keep="first")

        # Remove near-duplicates (same text after lowercasing and stripping)
        df["_text_norm"] = df["text"].str.lower().str.strip()
        df = df.drop_duplicates(subset=["_text_norm"], keep="first")
        df = df.drop(columns=["_text_norm"])

        after = len(df)
        log.info(f"  Removed {before - after} duplicates ({before} → {after})")

        # ── Step 4: Statistics ────────────────────────────────────────────
        log.info("\n" + "=" * 60)
        log.info("STEP 4: Dataset statistics")
        log.info("=" * 60)

        self._print_stats(df)

        # ── Step 5: Stratified split ──────────────────────────────────────
        log.info("\n" + "=" * 60)
        log.info("STEP 5: Train / Validation / Test split")
        log.info("=" * 60)

        train_df, temp_df = train_test_split(
            df, test_size=0.30, random_state=42, stratify=df["label_category"]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_category"]
        )

        log.info(f"  Train: {len(train_df)}")
        log.info(f"  Val:   {len(val_df)}")
        log.info(f"  Test:  {len(test_df)}")

        # ── Step 6: Export ────────────────────────────────────────────────
        log.info("\n" + "=" * 60)
        log.info("STEP 6: Exporting")
        log.info("=" * 60)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            # Parquet (efficient, typed)
            pq_path = output_path / f"{split_name}.parquet"
            split_df.to_parquet(pq_path, index=False)

            # JSONL (human-readable, HuggingFace compatible)
            jsonl_path = output_path / f"{split_name}.jsonl"
            split_df.to_json(jsonl_path, orient="records", lines=True)

            log.info(f"  {split_name}: {pq_path} + {jsonl_path}")

        # Export full dataset too
        df.to_parquet(output_path / "full.parquet", index=False)

        # Export statistics
        stats = self._compute_stats(df, train_df, val_df, test_df)
        with open(output_path / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        log.info(f"\n  Stats: {output_path / 'stats.json'}")
        log.info(f"\n{'=' * 60}")
        log.info(f"DONE! Dataset saved to {output_path}")
        log.info(f"{'=' * 60}")

        return df

    def _print_stats(self, df: pd.DataFrame):
        """Print dataset statistics."""
        log.info(f"\n  Total samples: {len(df)}")
        log.info("\n  Binary distribution:")
        for label, count in df["label_binary"].value_counts().items():
            pct = count / len(df) * 100
            label_name = "ATTACK" if label == 1 else "BENIGN"
            log.info(f"    {label_name}: {count} ({pct:.1f}%)")

        log.info("\n  Category distribution:")
        for cat, count in df["label_category"].value_counts().items():
            pct = count / len(df) * 100
            log.info(f"    {cat}: {count} ({pct:.1f}%)")

        log.info("\n  Source distribution:")
        for src, count in df["source"].value_counts().head(15).items():
            log.info(f"    {src}: {count}")

        log.info("\n  Languages:")
        for lang, count in df["language"].value_counts().head(10).items():
            log.info(f"    {lang}: {count}")

        log.info("\n  Text length stats:")
        lengths = df["text"].str.len()
        log.info(f"    Mean: {lengths.mean():.0f} chars")
        log.info(f"    Median: {lengths.median():.0f} chars")
        log.info(f"    Min: {lengths.min()} | Max: {lengths.max()}")

    def _compute_stats(self, df, train_df, val_df, test_df) -> dict:
        """Compute exportable statistics."""
        return {
            "total_samples": len(df),
            "splits": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            },
            "binary_distribution": df["label_binary"].value_counts().to_dict(),
            "category_distribution": df["label_category"].value_counts().to_dict(),
            "source_distribution": df["source"].value_counts().to_dict(),
            "language_distribution": df["language"].value_counts().to_dict(),
            "text_length": {
                "mean": float(df["text"].str.len().mean()),
                "median": float(df["text"].str.len().median()),
                "min": int(df["text"].str.len().min()),
                "max": int(df["text"].str.len().max()),
            },
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ShieldLM Dataset Curation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (requires HuggingFace access):
  python curate_dataset.py --output ./data/unified

  # InjecAgent only (works offline):
  python curate_dataset.py --output ./data/unified --skip-hf

  # Skip specific sources:
  python curate_dataset.py --output ./data/unified --skip multilingual safeguard
        """,
    )
    parser.add_argument("--output", default="./data/unified", help="Output directory")
    parser.add_argument("--injecagent", default="./InjecAgent", help="Path to InjecAgent repo")
    parser.add_argument(
        "--ahsanayub", default="./data/raw/ahsanayub-malicious-prompts",
        help="Path to ahsanayub/malicious-prompts clone (contains Harelix rows)",
    )
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace datasets")
    parser.add_argument("--skip", nargs="*", default=[], help="Source names to skip")
    parser.add_argument("--benign-samples", type=int, default=5000, help="Max benign samples")
    args = parser.parse_args()

    pipeline = DatasetPipeline(
        injecagent_path=args.injecagent,
        ahsanayub_path=args.ahsanayub,
        benign_sample_size=args.benign_samples,
    )
    pipeline.run(
        output_dir=args.output,
        skip_hf=args.skip_hf,
        skip_sources=args.skip,
    )


if __name__ == "__main__":
    main()
