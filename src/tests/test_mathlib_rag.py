import json
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.app.generator.model import RetrievalAugmentedGenerator, HTPSVerifiabilityEstimator
from src.app.prover.proof_search import HyperTreeProofSearchProver
from src.app.constants import RESULT_DATA_PATH, TRAIN_DATA_PATH, DATA_DIR, CORPUS_FILENAME, LEANDOJO_BENCHMARK
from src.app.data_extraction.extract_data import get_theorems_from_repo_url, convert_train_data_dicts
import sys


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"We use {device} as a device.")
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    # corpus_nameを指定するとdata/にretrieverで必要なコーパスの入ったベンチマークフォルダが作られます。RAGしないときはcorpus_nameは必要ありません。
    repo, theorems, positions = get_theorems_from_repo_url(
        "https://github.com/leanprover-community/mathlib",
        "19c869efa56bbb8b500f2724c0b77261edbfa28c",
        LEANDOJO_BENCHMARK
        )

    curriculum = list(zip(theorems, positions))

    tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean3-retriever-tacgen-byt5-small")
    base_generator = AutoModelForSeq2SeqLM.from_pretrained("kaiyuy/leandojo-lean3-retriever-tacgen-byt5-small")

    generator = RetrievalAugmentedGenerator(
        base_generator,
        tokenizer,
        warmup_steps=100,
        num_beams=1,
        eval_num_retrieved=1,
        eval_num_cpus=1,
        eval_num_theorems=1,
        max_seq_len=512,
        retriever_name="kaiyuy/leandojo-lean3-retriever-byt5-small",
        corpus_path=f"{DATA_DIR}/{LEANDOJO_BENCHMARK}/{CORPUS_FILENAME}"
        )
    estimator = HTPSVerifiabilityEstimator(base_generator, tokenizer, max_seq_len=512)
    prover = HyperTreeProofSearchProver(
        generator, 
        estimator,
        temperature=0.0,
        timeout=480,
        num_sampled_tactics=15
    )
    
    results = []
    for thm_pos in curriculum[:2]:
        theorem, position = thm_pos
        result = prover.search(repo, theorem, position)
        results.append(result)
    
    results_list = [result.to_dict() for result in results if result is not None]

    with open(RESULT_DATA_PATH, "w") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=4)
        
    train_data_dicts = convert_train_data_dicts(results)
    with open(TRAIN_DATA_PATH, "w") as f:
        json.dump(train_data_dicts, f, ensure_ascii=False, indent=4)
              
if __name__ == '__main__':
    main()