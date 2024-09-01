import json

from src.app.generator.model import *
from src.app.prover.proof_search import *
from src.app.data_extraction.extract_data import get_theorems_from_repo_url
from src.app.prover.setup_model import setup_model

from src.app.constants import VISUALIZED_DATA_PATH, DATA_DIR

repo, theorems, positions = get_theorems_from_repo_url("https://github.com/tukamilano/lean3-example", "88cff6db83fefd5565efc4e54937efd6b9e14d7c")
curriculum = list(zip(theorems, positions))
theorem, position = curriculum[0]

# setup_model()はget_theorems_from_repo_urlの後ろにつけるか、あらかじめコーパスが存在する状態で実行してください。
_, _, prover, _ = setup_model(benchmark="tukamilano-lean3-example")
result = prover.search(repo, theorem, position)

with open(VISUALIZED_DATA_PATH, "w") as f:
    json.dump([node.to_dict() for node in result.hypergraph.values()], f, ensure_ascii=False, indent=2)
    print(result.proof)
