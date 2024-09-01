from src.app.generator.model import *
from src.app.prover.proof_search import *
from src.app.data_extraction.extract_data import get_theorems_from_repo_url, convert_train_data_dicts
from src.app.prover.setup_model import setup_model

repo, theorems, positions = get_theorems_from_repo_url("https://github.com/tukamilano/lean3-example", "88cff6db83fefd5565efc4e54937efd6b9e14d7c")
curriculum = list(zip(theorems, positions))
theorem, position = curriculum[0]

_, _, prover, _ = setup_model(benchmark="tukamilano-lean3-example")
result = prover.search(repo, theorem, position)

result_dicts = convert_train_data_dicts([result])
print(result_dicts)
