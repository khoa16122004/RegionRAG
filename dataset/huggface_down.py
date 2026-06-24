"""Download all VisRAG evaluation datasets from Hugging Face.

Example:
	python dataset/huggface_down.py
	python dataset/huggface_down.py --out_dir data/VisRAG
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


VISRAG_EVAL_REPOS = [
	"openbmb/VisRAG-Ret-Test-ArxivQA",
	"openbmb/VisRAG-Ret-Test-ChartQA",
	"openbmb/VisRAG-Ret-Test-InfoVQA",
	"openbmb/VisRAG-Ret-Test-MP-DocVQA",
	"openbmb/VisRAG-Ret-Test-PlotQA",
	"openbmb/VisRAG-Ret-Test-SlideVQA",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Download all VisRAG eval datasets into one local folder."
	)
	parser.add_argument(
		"--out_dir",
		type=Path,
		default=Path("data_dir") / "VisRAG",
		help="Root output directory. Each dataset will be stored under this folder.",
	)
	parser.add_argument(
		"--token",
		type=str,
		default=None,
		help="Hugging Face token (optional).",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Force re-download even if files already exist.",
	)
	return parser.parse_args()


def download_all_visrag_eval(out_dir: Path, token: str | None = None, force: bool = False) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)
	print(f"[INFO] Saving VisRAG eval datasets to: {out_dir.resolve()}")

	for repo_id in VISRAG_EVAL_REPOS:
		dataset_name = repo_id.split("/")[-1]
		target_dir = out_dir / dataset_name
		print(f"\n[INFO] Downloading {repo_id} -> {target_dir}")

		snapshot_download(
			repo_id=repo_id,
			repo_type="dataset",
			local_dir=str(target_dir),
			token=token,
			local_files_only=False,
			force_download=force,
			resume_download=not force,
		)

	print("\n[DONE] All VisRAG eval datasets downloaded.")


if __name__ == "__main__":
	args = parse_args()
	download_all_visrag_eval(out_dir=args.out_dir, token=args.token, force=args.force)