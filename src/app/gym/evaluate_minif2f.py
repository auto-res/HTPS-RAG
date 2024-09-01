from math import ceil
from typing import List, Tuple
from loguru import logger
import json

from lean_dojo import Theorem, Pos, LeanGitRepo, TracedTheorem, trace
from src.app.prover.hyper_tree import Status
from app.generator.model import RetrievalAugmentedGenerator, HTPSVerifiabilityEstimator
from app.prover.proof_search import HyperTreeProofSearchProver
from app.prover.setup_model import setup_model
from src.app.constants import TEST_DATA_PATH, EPOCH_SIZE, BATCH_SIZE, TRY_COUNT_LIMIT, DATA_DIR, MODEL_PATH, OPTIMIZER_PATH, CRITIC_LINEAR_PATH, CRITIC_OPTIMIZER_PATH, MINIF2F_BENCHMARK
from src.app.retrieval.export_benchmark import split_data, export_benchmark

def main():
    minif2f = LeanGitRepo(
        "https://github.com/facebookresearch/miniF2F",
        "5271ddec788677c815cf818a06f368ef6498a106",
    )
    traced_minif2f = trace(minif2f)

    benchmark_dir = f"{DATA_DIR}/{MINIF2F_BENCHMARK}"
    splits = split_data(traced_minif2f)
    export_benchmark(traced_minif2f, splits, benchmark_dir)

    _, _, prover, _ = setup_model(
        model_path="kaiyuy/leandojo-lean3-tacgen-byt5-small",
        tokenizer_path="kaiyuy/leandojo-lean3-tacgen-byt5-small",
        retriever_name="kaiyuy/leandojo-lean3-retriever-byt5-small",
        benchmark=MINIF2F_BENCHMARK,
        saved_model_path=MODEL_PATH,
        optimizer_path=OPTIMIZER_PATH,
        critic_linear_path=CRITIC_LINEAR_PATH,
        critic_optimizer_path=CRITIC_OPTIMIZER_PATH
        )

    splits = {"default": {"val": [], "test": []}}

    for tf in traced_minif2f.get_traced_theorems():
        if tf.repo.name != "miniF2F":
            continue
        if tf.file_path.name == "valid.lean":
            splits["default"]["val"].append(tf)
        else:
            assert tf.file_path.name == "test.lean"
            splits["default"]["test"].append(tf)

    test_curriculum = splits["default"]["test"]
    block_count = ceil(len(test_curriculum)/BATCH_SIZE)
    with open(TEST_DATA_PATH, 'w') as f:
        for _ in range(EPOCH_SIZE):
            for block_num in range(block_count):
                target_curriculum = test_curriculum[block_num*BATCH_SIZE:(block_num + 1)*BATCH_SIZE]
                theorems, positions = _convert(target_curriculum)
                test_results = prover.batch_search_by_passn(repo=minif2f, theorems=theorems, positions=positions, try_count_limit=TRY_COUNT_LIMIT)

                results_dicts = [result.to_dict() for result in test_results if result is not None]
                json.dump(results_dicts, f, ensure_ascii=False, indent=4)

            # Calculate the result statistics.
            num_proved = num_failed = num_discarded = 0
            for r in test_results:
                if r is None:
                    num_discarded += 1
                elif r.status == Status.PROVED:
                    num_proved += 1
                else:
                    num_failed += 1

            logger.info(
                f"Evaluation done! {num_proved} theorems proved, {num_failed} theorems failed, {num_discarded} non-theorems discarded"
            )
            logger.info(f"Pass@{TRY_COUNT_LIMIT}: {num_proved / (num_proved + num_failed)}")

def _convert(curriculum: List[TracedTheorem]) -> Tuple[List[Theorem], List[Pos]]:
    theorems, positions = [], []
    for tf in curriculum:
        theorems.append(tf.theorem)
        positions.append(tf.start)

    return theorems, positions

if __name__ == '__main__':
    main()
