import sys

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from lean_dojo import *

from src.app.generator.model import *
from src.app.prover.proof_search import *
from src.app.data_extraction.extract_data import get_theorems_from_repo_url
from lean_dojo import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.app.constants import NUM_SAMPLED_TACTICS

def main():
    tokenizer_path = "kaiyuy/leandojo-lean3-tacgen-byt5-small"
    model_path = "kaiyuy/leandojo-lean3-tacgen-byt5-small"
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    base_generator = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    base_generator.eval()

    generator = RetrievalAugmentedGenerator(
        base_generator,
        tokenizer,
        num_beams=1,
        eval_num_retrieved=1,
        eval_num_cpus=1,
        eval_num_theorems=1,
        max_seq_len=512
        )
    estimator = HTPSVerifiabilityEstimator(
        base_generator,
        tokenizer,
        max_seq_len=512
        )

    prover = HyperTreeProofSearchProver(
        generator,
        estimator,
        timeout=10,
        num_sampled_tactics=NUM_SAMPLED_TACTICS,
    )

    repo, theorems, positions = get_theorems_from_repo_url("https://github.com/tukamilano/lean3-example", "88cff6db83fefd5565efc4e54937efd6b9e14d7c")

    results = prover.batch_search(repo, theorems, positions)

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    main()
