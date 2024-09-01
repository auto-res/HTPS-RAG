import random
from math import ceil
from typing import List, Tuple
from loguru import logger
import time

from lean_dojo import Theorem, Pos
from src.app.data_extraction.extract_data import convert_train_data_dicts, get_theorems_from_repo_url
from src.app.data_extraction.training_history import *
from app.prover.trainer import train_model
from app.prover.hyper_tree import Status
from app.prover.proof_search import SearchResult
from src.app.prover.setup_model import setup_model
from src.app.constants import TRAIN_DATA_PATH, BATCH_SIZE, LEANDOJO_BENCHMARK, MODEL_PATH, OPTIMIZER_PATH, CRITIC_LINEAR_PATH, CRITIC_OPTIMIZER_PATH
from src.app.common import append_to_file

def main():
    VALID_SIZE = 100
    TRAINING_DURATION = 60 * 60 * 20 # 20 hours
    VALIDATION_DURATION = 60 * 60 * 4 - 1800 # 3.5 hours
    start_time = time.time()
    repo, theorems, positions = get_theorems_from_repo_url("https://github.com/leanprover-community/mathlib", "ce64cd319bb6b3e82f31c2d38e79080d377be451", LEANDOJO_BENCHMARK)
    
    curriculum = list(zip(theorems, positions))
    
    # split curriculum into training and validation sets
    random.seed(42)
    random.shuffle(curriculum)
    train_curriculum, valid_curriculum = curriculum[:-VALID_SIZE], curriculum[-VALID_SIZE:]
    
    block_count = ceil(len(train_curriculum)/BATCH_SIZE)

    generator, estimator, prover, tokenizer = setup_model(
        model_path="kaiyuy/leandojo-lean3-tacgen-byt5-small", 
        tokenizer_path="kaiyuy/leandojo-lean3-tacgen-byt5-small", 
        retriever_name="kaiyuy/leandojo-lean3-retriever-byt5-small", 
        benchmark=LEANDOJO_BENCHMARK,
        saved_model_path=MODEL_PATH, 
        optimizer_path=OPTIMIZER_PATH, 
        critic_linear_path=CRITIC_LINEAR_PATH, 
        critic_optimizer_path=CRITIC_OPTIMIZER_PATH
        )
    
    # Clear the training data file.
    with open(TRAIN_DATA_PATH, 'w') as f:
        pass
    
    acc_history = AccuracyHistory.from_checkpoint()
    loss_history = LossHistory.from_checkpoint()
    block_start = len(loss_history)
    for block_num in range(block_start, block_count):
        target_curriculum = train_curriculum[block_num*BATCH_SIZE:(block_num + 1)*BATCH_SIZE]
        theorems, positions = _convert(target_curriculum)
        results = prover.batch_search(repo=repo, theorems=theorems, positions=positions)
        train_data_dict = convert_train_data_dicts(results)
        append_to_file(TRAIN_DATA_PATH, train_data_dict)

        # Update the model.
        gen_loss, est_loss = train_model(generator, estimator, tokenizer)
        loss_history.add(gen_loss, est_loss)
        acc_history.add(get_accuracy(results))
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > TRAINING_DURATION:
            loss_history.save()
            acc_history.save()
            break

    current_time = time.time()
    valid_results = []
    for thm, pos in valid_curriculum:
        result = prover.search(repo, thm, pos)
        valid_results.append(result)
        elapsed_time = time.time() - start_time
        if elapsed_time > (TRAINING_DURATION + VALIDATION_DURATION):
            break
    
    # Calculate the result statistics.
    accuracy = get_accuracy(valid_results)
    logger.info(f"Validation Accuracy: {accuracy}")
    loss_history.render()
    loss_history.save()
    
def get_accuracy(results: List[SearchResult]) -> float:
    num_proved = num_failed = num_discarded = 0
    for r in results:
        if r is None:
            num_discarded += 1
        elif r.status == Status.PROVED:
            num_proved += 1
        else:
            num_failed += 1

    return num_proved / (num_proved + num_failed)

def _convert(curriculum: List[Tuple[Theorem, Pos]]) -> Tuple[List[Theorem], List[Pos]]:
    theorems, positions = zip(*curriculum)
    
    return list(theorems), list(positions)

if __name__ == '__main__':
    start = time.time()
    main()

    elapsed_time = time.time() - start

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    print(f"It took {hours} hours, {minutes} minutes, and {seconds:.2f} seconds to run.")