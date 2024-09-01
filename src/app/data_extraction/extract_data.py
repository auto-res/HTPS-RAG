from typing import Dict, Union, List, Any, Tuple, Optional
from itertools import chain

from src.app.prover.proof_search import SearchResult, InternalNode
from lean_dojo import LeanGitRepo, Theorem, Pos, TracedRepo, get_traced_repo_path
from src.app.retrieval.export_benchmark import split_data, export_benchmark
from src.app.constants import DATA_DIR
from pathlib import Path

def get_theorems_from_repo_url(url: str, commit: str, benchmark_name:str=None) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    repo = LeanGitRepo(url, commit)
    traced_repo_path = get_traced_repo_path(repo)
    traced_repo = TracedRepo.load_from_disk(traced_repo_path)
    traced_theorems = traced_repo.get_traced_theorems()
    traced_theorems = list(filter(lambda traced_theorem: traced_theorem.repo == repo, traced_theorems))
    theorems = [traced_theorem.theorem for traced_theorem in traced_theorems]
    positions = [traced_theorem.start for traced_theorem in traced_theorems]
    # export corpus for RAG
    if benchmark_name is not None:
        benchmark_dir = Path(f"{DATA_DIR}/{benchmark_name}")
        splits = split_data(traced_repo)
        export_benchmark(traced_repo, splits, benchmark_dir)
    return repo, theorems, positions

def convert_train_data_dicts(results: List[Optional[SearchResult]]) -> list:
    train_data_dicts = list(chain.from_iterable(_result_to_dicts(result) for result in results if result is not None))
    return train_data_dicts

def _result_to_dicts(result: SearchResult) -> List[Dict[str, Union[str, List[Dict[str, Any]], float]]]:
    return [_node_to_dict(node) for node in result.hypergraph.nodes if node.is_explored]

def _node_to_dict(node:InternalNode) -> Dict[str, Union[str, Dict[str, Dict[str, Union[int, bool]]], float]]:
    assert node.is_explored
    return {
            "state": node.goal.pp,
            "tactics_info": {
                edge.tactic: {
                    "visit_count": edge.visit_count,
                    "is_valid": edge.action_value > 0,
                    "is_solved": edge.action_value == 1.0
                }
                for edge in node.out_edges
            },
            "parent_visit_count": node.visit_count,
            "verifiability": node.verifiability
        }
