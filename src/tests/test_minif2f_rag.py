from lean_dojo import LeanGitRepo, trace
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.app.generator.model import *
from src.app.prover.proof_search import *
from src.app.constants import DATA_DIR, CORPUS_FILENAME, MINIF2F_BENCHMARK
from src.app.retrieval.export_benchmark import split_data, export_benchmark

minif2f = LeanGitRepo(
    "https://github.com/facebookresearch/miniF2F",
    "5271ddec788677c815cf818a06f368ef6498a106",
)
traced_minif2f = trace(minif2f)

# コーパスを含むベンチマークフォルダの生成（setup_modelに同様の処理あり）
benchmark_dir = f"{DATA_DIR}/{MINIF2F_BENCHMARK}"
splits = split_data(traced_minif2f)
export_benchmark(traced_minif2f, splits, benchmark_dir)

tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean3-tacgen-byt5-small")
base_generator = AutoModelForSeq2SeqLM.from_pretrained("kaiyuy/leandojo-lean3-tacgen-byt5-small")
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
    corpus_path=f"{benchmark_dir}/{CORPUS_FILENAME}"
    )
estimator = HTPSVerifiabilityEstimator(base_generator, tokenizer, max_seq_len=512)
prover = HyperTreeProofSearchProver(
    generator, 
    estimator,
    timeout=10,
    num_sampled_tactics=10
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