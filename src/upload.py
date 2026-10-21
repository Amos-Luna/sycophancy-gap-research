import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = ("turn1", "turn2", "discrimination", "confidence")


def dataset_card(
    results: Path
    ) -> str:
    """Builds a README with YAML metadata for publishing the dataset on the Hugging Face Hub.

    Includes the questions configuration and each run's raw result files.

    Args:
        results: Path to the directory containing experiment results.

    Returns:
        The README content with YAML metadata and the project's README.
    """
    runs = sorted(d.name for d in results.iterdir() if (d / "meta.json").exists())
    configs = ["- config_name: questions\n  data_files: data/questions.jsonl"]
    configs += [f"- config_name: {run.replace('.', '_')}-{name}\n  data_files: results/{run}/{name}.jsonl"
                for run in runs for name in RAW_FILES]
    metadata = ("---\nlicense: apache-2.0\nlanguage:\n- en\n- es\npretty_name: Sycophancy under pushback, English vs Spanish\n"
                "tags:\n- sycophancy\n- alignment\n- evaluation\n- multilingual\nsize_categories:\n- 1K<n<10K\n"
                "configs:\n" + "\n".join(configs) + "\n---\n\n")
    return metadata + (ROOT / "README.md").read_text(encoding="utf-8")


def evaluated_models(
    results: Path
    ) -> set[tuple[str, str]]:
    """Collects the model IDs and revisions evaluated in the results directory.
    (model id, commit) of every run in the results folder.

    Args:
        results: Path to the directory containing experiment results.

    Returns:
        A set of (model ID, commit revision) pairs.
    """
    metas = (json.loads(path.read_text()) for path in results.glob("*/meta.json"))
    return {(meta["model"], meta["revision"]) for meta in metas}


def main() -> None:
    """Publishes the dataset, experiment results, and code to the Hugging Face Hub.

    Creates a dataset repository and a collection, adds evaluated models to the
    collection, and optionally uploads copies of their model weights.

    Command-line arguments configure the repository, results directory,
    model-weight copying, and repository visibility.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="<user>/<name> of the dataset repo")
    parser.add_argument("--results", default=str(ROOT / "results"), help="results folder to publish")
    parser.add_argument("--copy-models", action="store_true", help="also upload a copy of the model weights")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api, results = HfApi(), Path(args.results)
    user = api.whoami()["name"]
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=dataset_card(results).encode(), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")
    api.upload_folder(folder_path=ROOT, repo_id=args.repo, repo_type="dataset", commit_message="Code and data",
                      allow_patterns=["data/*", "src/*.py", "notebooks/*.ipynb"])
    api.upload_folder(folder_path=results, path_in_repo="results", repo_id=args.repo, repo_type="dataset",
                      commit_message="Results")

    collection = api.create_collection(title="Sycophancy under pushback: English vs Spanish", exists_ok=True,
                                       description="Dataset, results, code and the models evaluated.")
    api.add_collection_item(collection.slug, args.repo, "dataset", exists_ok=True)
    for model, revision in sorted(evaluated_models(results)):
        api.add_collection_item(collection.slug, model, "model", note=f"Evaluated at commit {revision}", exists_ok=True)
        if args.copy_models:
            copy = f"{user}/{model.split('/')[-1]}-{revision[:8]}"
            api.create_repo(copy, repo_type="model", private=args.private, exist_ok=True)
            api.upload_folder(folder_path=snapshot_download(model, revision=revision), repo_id=copy, repo_type="model")
            api.add_collection_item(collection.slug, copy, "model", exists_ok=True)
            print("model copy:", f"https://huggingface.co/{copy}")

    print("dataset:   ", f"https://huggingface.co/datasets/{args.repo}")
    print("collection:", f"https://huggingface.co/collections/{collection.slug}")


if __name__ == "__main__":
    main()
