import json
import shutil
import random
import networkx as nx
from copy import copy
from pathlib import Path
from loguru import logger
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Union

import lean_dojo
from lean_dojo import *
from lean_dojo.constants import LEAN3_DEPS_DIR
from app.constants import DATA_DIR

# from https://github.com/lean-dojo/LeanDojo/blob/main/scripts/generate-benchmark-lean3.ipynb
# https://github.com/lean-dojo/LeanDojo/blob/main/scripts/generate-benchmark-lean4.ipynb

random.seed(3407)  # https://arxiv.org/abs/2109.08203

NUM_VAL = NUM_TEST = 2000

SPLIT_NAME = str  # train/val/test
SPLIT = Dict[SPLIT_NAME, List[TracedTheorem]]
SPLIT_STRATEGY = str

def _split_sequentially(
    traced_theorems: List[TracedTheorem],
) -> SPLIT:
    """Split ``traced_theorems`` sequentially into train/val/test."""
    num_theorems = len(traced_theorems)
    num_train = num_theorems - NUM_VAL - NUM_TEST
    return {
        "train": traced_theorems[:num_train],
        "val": traced_theorems[num_train : num_train + NUM_VAL],
        "test": traced_theorems[num_train + NUM_VAL :],
    }


def split_randomly(
    traced_theorems: List[TracedTheorem],
) -> SPLIT:
    """Split ``traced_theorems`` randomly into train/val/test."""
    logger.info("Splitting the theorems randomly")
    traced_theorems = copy(traced_theorems)
    random.shuffle(traced_theorems)
    return _split_sequentially(traced_theorems)

def _split_sequentially(
    traced_theorems: List[TracedTheorem],
) -> SPLIT:
    """Split ``traced_theorems`` sequentially into train/val/test."""
    num_theorems = len(traced_theorems)
    num_train = num_theorems - NUM_VAL - NUM_TEST
    return {
        "train": traced_theorems[:num_train],
        "val": traced_theorems[num_train : num_train + NUM_VAL],
        "test": traced_theorems[num_train + NUM_VAL :],
    }


def split_by_premise(
    traced_theorems: List[TracedTheorem],
) -> SPLIT:
    """
    Split theorems into train/val/test so that proofs in val/test rely on at
    least one novel premise that does not appear in train.
    """
    logger.info("Splitting the theorems by premises")

    # Figure out the number of theorems in train/val/test.
    num_theorems = len(traced_theorems)
    num_val_test = NUM_VAL + NUM_TEST
    theorems_val_test = set()

    # Map each premise to a list of theorems using it.
    theorems_by_premises = defaultdict(list)
    for t in traced_theorems:
        for p in t.get_premise_full_names():
            theorems_by_premises[p].append(t)

    # Sort the premises by the number of theorems using them (in ascending order).
    theorems_by_premises = sorted(theorems_by_premises.items(), key=lambda x: len(x[1]))

    # For each premise, put all theorems using it into val_test so that it does not appear in train.
    for _, thms in theorems_by_premises:
        if len(theorems_val_test) < num_val_test:
            theorems_val_test.update(thms)

    # All other theorems go to train.
    theorems_train = [t for t in traced_theorems if t not in theorems_val_test]
    theorems_val_test = list(theorems_val_test)
    random.shuffle(theorems_val_test)

    return {
        "train": theorems_train,
        "val": theorems_val_test[:NUM_VAL],
        "test": theorems_val_test[NUM_VAL:],
    }

def split_data(traced_repo: TracedRepo) -> Dict[SPLIT_STRATEGY, SPLIT]:
    traced_theorems = traced_repo.get_traced_theorems()
    logger.info(f"{len(traced_theorems)} theorems in total")

    return {
        "random": split_randomly(traced_theorems),
        "novel_premises": split_by_premise(traced_theorems),
    }

def export_proofs(
    traced_repo: TracedRepo, splits: Dict[SPLIT_STRATEGY, SPLIT], dst_path: Path
) -> None:
    """Export all proofs in a traced repo to ``dst_path''."""
    for strategy, split in splits.items():
        split_dir = dst_path / strategy
        split_dir.mkdir(parents=True)

        for name, theorems in split.items():
            data = []
            num_tactics = 0

            for thm in theorems:
                tactics = [
                    {
                        "tactic": t.tactic,
                        "annotated_tactic": t.get_annotated_tactic(),
                        "state_before": t.state_before,
                        "state_after": t.state_after,
                    }
                    for t in thm.get_traced_tactics()
                    if t.state_before != "no goals"
                    and "·" not in t.tactic  # Ignore "·".
                ]
                num_tactics += len(tactics)
                data.append(
                    {
                        "url": thm.repo.url,
                        "commit": thm.repo.commit,
                        "file_path": str(thm.theorem.file_path),
                        "full_name": thm.theorem.full_name,
                        "start": list(thm.start),
                        "end": list(thm.end),
                        "traced_tactics": tactics,
                    }
                )
            oup_path = split_dir / f"{name}.json"
            json.dump(data, oup_path.open("wt"))
            logger.info(
                f"{len(theorems)} theorems and {num_tactics} tactics saved to {oup_path}"
            )


def export_premises(traced_repo: TracedRepo, dst_path: Path) -> None:
    """Export all premise definitions in a traced repo to ``dst_path``."""
    oup_path = dst_path / "corpus.jsonl"
    num_premises = 0

    with oup_path.open("wt") as oup:
        G = traced_repo.traced_files_graph

        for tf_node in reversed(list(nx.topological_sort(G))):
            tf = G.nodes[tf_node]["traced_file"]
            imports = [str(_) for _ in G.successors(tf_node)]
            premises = tf.get_premise_definitions()
            num_premises += len(premises)
            oup.write(
                json.dumps(
                    {"path": str(tf.path), "imports": imports, "premises": premises}
                )
                + "\n"
            )
    logger.info(
        f"{num_premises} theorems/definitions from {traced_repo.num_traced_files} files saved to {oup_path}"
    )


def export_licenses(traced_repo: TracedRepo, dst_path: Path) -> None:
    """Export the licenses of a traced repo and all its dependencies to ``dst_path``."""
    license_dir = dst_path / "licenses"
    license_dir.mkdir()
    all_repos = [traced_repo.repo] + list(traced_repo.dependencies.values())

    for repo in all_repos:
        lic = repo.get_license()
        if lic is None:
            continue
        with (license_dir / repo.name).open("wt") as oup:
            oup.write(lic)

    with (license_dir / "README.md").open("wt") as oup:
        oup.write(
            "This directory contains licenses of Lean repos used to generate this dataset. The dataset itself is released under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)."
        )


def export_metadata(traced_repo: TracedRepo, dst_path: Path, **kwargs) -> None:
    """Export the metadata of a traced repo to ``dst_path''."""
    metadata = dict(kwargs)
    metadata["creation_time"] = str(datetime.now())
    metadata["from_repo"] = {
        "url": traced_repo.repo.url,
        "commit": traced_repo.repo.commit,
    }
    metadata["leandojo_version"] = lean_dojo.__version__
    json.dump(metadata, (dst_path / "metadata.json").open("wt"))

def export_benchmark(
    traced_repo: TracedRepo,
    splits: Dict[SPLIT_STRATEGY, SPLIT],
    dst_path: Union[str, Path],
    **kwargs,
) -> None:
    """Export a traced repo whose theorems have been splitted to ``dst_path``."""
    if isinstance(dst_path, str):
        dst_path = Path(dst_path)
    if dst_path.exists():
        logger.warning(f"{dst_path} already exists. Removing it now.")
        shutil.rmtree(dst_path)

    # Export the proofs.
    export_proofs(traced_repo, splits, dst_path)

    # Export the premises (theorems, definitions, etc.).
    export_premises(traced_repo, dst_path)

    # Export the licenses.
    export_licenses(traced_repo, dst_path)

    # Export metadata.
    export_metadata(traced_repo, dst_path, **kwargs)

def main(url:str=None, commit:str=None, corpus_name:str=None):
    DST_DIR = Path(f"{DATA_DIR}/{corpus_name}")
    repo = LeanGitRepo(url, commit)
    traced_repo = trace(repo)
    splits = split_data(traced_repo)
    export_benchmark(traced_repo, splits, DST_DIR, dataset_name="LeanDojo Benchmark 4")