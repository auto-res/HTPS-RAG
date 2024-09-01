import argparse

from lean_dojo import LeanGitRepo, trace
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.app.generator.model import *
from src.app.prover.proof_search import *
from src.app.constants import NUM_SAMPLED_TACTICS

minif2f = LeanGitRepo(
    "https://github.com/facebookresearch/miniF2F",
    "5271ddec788677c815cf818a06f368ef6498a106",
)
traced_minif2f = trace(minif2f)

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
    timeout=1000,
    num_sampled_tactics=NUM_SAMPLED_TACTICS,
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

assert len(splits["default"]["val"]) > 0
assert len(splits["default"]["test"]) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_name", type=str, default="mathd_numbertheory_1124")
    args = parser.parse_args()
    
    tf = list(filter(lambda x: x.theorem.full_name == args.full_name, traced_minif2f.get_traced_theorems()))[0]
    prover.search(tf.repo, tf.theorem, tf.start)