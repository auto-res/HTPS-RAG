from sklearn.model_selection import train_test_split
from math import ceil
from typing import List, Tuple
from loguru import logger
import time

from lean_dojo import Theorem, Pos
from src.app.data_extraction.extract_data import convert_train_data_dicts, get_theorems_from_repo_url
from src.app.data_extraction.training_history import AccuracyHistory
from app.prover.trainer import train_model
from app.prover.setup_model import setup_model
from app.prover.hyper_tree import Status
from src.app.constants import (
    TRAIN_DATA_PATH, 
    TRY_COUNT_LIMIT,
    DATA_DIR
)
from src.app.common import append_to_file

from loguru import logger

def main():
    repo, theorems, positions = get_theorems_from_repo_url(
        "https://github.com/tukamilano/lean3-example",
        "88cff6db83fefd5565efc4e54937efd6b9e14d7c",
        "tukamilano-lean3-example"
        )

    curriculum = list(zip(theorems, positions))
    
    train_curriculum, valid_curriculum = train_test_split(curriculum[:4], test_size=0.2)
    
    block_count = ceil(len(train_curriculum)/1)

    generator, estimator, prover, tokenizer = setup_model(benchmark="tukamilano-lean3-example")
    prover.timeout = 5

    with open(TRAIN_DATA_PATH, 'w') as f:
        pass

    history = AccuracyHistory(x_label='Epoch', y_label='Accuracy', title='tukamilano accuracy', fig_name='tukamilano_accuracy.png')
    for i in range(4):
        for block_num in range(block_count):
            target_curriculum = train_curriculum[block_num*1:(block_num + 1)*1]
            theorems, positions = _convert(target_curriculum)
            results = prover.batch_search(repo=repo, theorems=theorems, positions=positions)
            train_data_dict = convert_train_data_dicts(results)
            append_to_file(TRAIN_DATA_PATH, train_data_dict)

            # Update the model.
            logger.info(f"generator warmup level: {generator.warmup_level}")
            logger.info(f"estimator warmup level: {estimator.warmup_level}")
            train_model(generator, estimator, tokenizer)

        valid_theorems, valid_positions = _convert(valid_curriculum)

        try_count_limit = 1
        valid_results = prover.batch_search_by_passn(repo, theorems=valid_theorems, positions=valid_positions, try_count_limit=try_count_limit)
    
        # Calculate the result statistics.
        num_proved = num_failed = num_discarded = 0
        for r in valid_results:
            if r is None:
                num_discarded += 1
            elif r.status == Status.PROVED:
                num_proved += 1
            else:
                num_failed += 1

        accuracy = num_proved / (num_proved + num_failed)
        logger.info(
            f"Evaluation done! {num_proved} theorems proved, {num_failed} theorems failed, {num_discarded} non-theorems discarded"
        )
        logger.info(f"Pass@{TRY_COUNT_LIMIT}: {accuracy}")
        history.add(accuracy)
    
    history.render()

def _convert(curriculum: List[Tuple[Theorem, Pos]]) -> Tuple[List[Theorem], List[Pos]]:
    theorems, positions = zip(*curriculum)
    
    return list(theorems), list(positions)

if __name__ == '__main__':
    start_time = time.time()
    main()

    elapsed_time = time.time() - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    print(f"It took {hours} hours, {minutes} minutes, and {seconds:.2f} seconds to run.")
