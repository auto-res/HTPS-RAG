from src.app.generator.model import RLDataset, RetrievalAugmentedGenerator, HTPSVerifiabilityEstimator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from src.app.constants import MODEL_PATH, CRITIC_OPTIMIZER_PATH, CRITIC_LINEAR_PATH, SOLVED_WEIGHT, VALID_WEIGHT, INVALID_WEIGHT

#恩田さんが提案してくださったデータの形式
sample_data = [
    {
        "state": "a b c : ℕ\n⊢ a + b + c = a + c + b",
        "tactics_info": {
                "induction": {
                    "visit_count": 3,
                    "is_valid": False,
                    "is_solved": False
                },
                "add_assoc": {
                    "visit_count": 6,
                    "is_valid": False,
                    "is_solved": False                    
                }
            },    
        "parent_visit_count": 9,
        "verifiability": 1.0
    }
]

#DataLoaderに渡すためにデータを変換
transformed_data = []

for item in sample_data:
    state = item["state"]
    tactics_info = item["tactics_info"]
    
    for tactic, tactic_info in tactics_info.items():
        transformed_item = {
            "state": state,
            "tactic": tactic,
            "visit_count": tactic_info["visit_count"],
            "learning_weight": SOLVED_WEIGHT if tactic_info["is_solved"] else VALID_WEIGHT if tactic_info["is_valid"] else INVALID_WEIGHT,
            "parent_visit_count": item["parent_visit_count"],
            "verifiability": item["verifiability"]
        }
        transformed_data.append(transformed_item)

tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean3-tacgen-byt5-small")

transformed_data = RLDataset(transformed_data, tokenizer)
data = DataLoader(transformed_data, batch_size=32, shuffle=True)
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
    )
estimator = HTPSVerifiabilityEstimator(
    base_generator,
    tokenizer,
    warmup_steps=100,
    max_seq_len=512
    )

epochs = 1
for i in range(epochs):
    for batch in data:
        generator.train(batch)
        estimator.train(batch)
        estimator.save_model(MODEL_PATH, CRITIC_OPTIMIZER_PATH, CRITIC_LINEAR_PATH)