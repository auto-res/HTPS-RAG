from sklearn.model_selection import train_test_split
from math import ceil
from typing import Any, List, Tuple
from loguru import logger
import time
import optuna
import random

from lean_dojo import Theorem, Pos
from src.app.data_extraction.extract_data import convert_train_data_dicts, get_theorems_from_repo_url
from src.app.data_extraction.training_history import *
from app.prover.trainer import train_model
from app.prover.hyper_tree import Status
from app.prover.proof_search import SearchResult
from src.app.prover.setup_model import setup_model
from src.app.constants import TRAIN_DATA_PATH, EPOCH_SIZE, BATCH_SIZE, TRY_COUNT_LIMIT, LEANDOJO_BENCHMARK, OPTUNA_TIMEOUT, WARMUP_STEPS, EXPANSION_LIMIT
from src.app.common import append_to_file

VALID_SIZE = 100
TRAINING_DURATION = 60 * 45 # 45 min
VALIDATION_DURATION = 60 * 10 # 10 minutes

class Objective:
    def __init__(self):
        self.repo, theorems, positions = get_theorems_from_repo_url(
            "https://github.com/leanprover-community/mathlib", 
            "ce64cd319bb6b3e82f31c2d38e79080d377be451",
            )
        self.curriculum = list(zip(theorems, positions))
        
    def __call__(self, trial):
        start_time = time.time()
        # split curriculum into training and validation sets
        random.seed(trial.number)
        random.shuffle(self.curriculum)
        train_curriculum, valid_curriculum = self.curriculum[:-VALID_SIZE], self.curriculum[-VALID_SIZE:]
        
        block_count = ceil(len(train_curriculum)/BATCH_SIZE)

        generator_params = {
            'num_beams': trial.suggest_int('generator_num_beams', 1, 5),
            'eval_num_retrieved': trial.suggest_int('generator_eval_num_retrieved', 1, 10),
            'lr': trial.suggest_float('generator_lr', 0.0001, 0.01),
            'warmup_steps': WARMUP_STEPS,
        }

        estimator_params = {
            'warmup_steps': WARMUP_STEPS,
            'lr': trial.suggest_float('estimator_lr', 0.0001, 0.01),
        }

        prover_params = {
            'num_sampled_tactics': trial.suggest_int('prover_num_sampled_tactics', 5, 20),
            'temperature': trial.suggest_float('prover_temparture', 0.1, 5.0),
            'num_expansion': EXPANSION_LIMIT,
        }

        generator, estimator, prover, tokenizer = setup_model(
        benchmark=LEANDOJO_BENCHMARK,
        generator_config=generator_params,
        estimator_config=estimator_params,
        prover_config=prover_params,
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
            results = prover.batch_search(repo=self.repo, theorems=theorems, positions=positions)
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
            result = prover.search(self.repo, thm, pos)
            valid_results.append(result)
            elapsed_time = time.time() - start_time
            if elapsed_time > (TRAINING_DURATION + VALIDATION_DURATION):
                break
        
        accuracy = get_accuracy(valid_results)
        acc_history.add(accuracy)
        
        acc_history.render()
        loss_history.render()
        acc_history.save()
        loss_history.save()

        total_loss = loss_history[-1]["generator_loss"] + loss_history[-1]["estimator_loss"]
        
        return total_loss

def get_accuracy(results: list[SearchResult]) -> float:
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
    start_time = time.time()

    study = optuna.create_study(direction='minimize')

    objective = Objective()

    study.optimize(objective, timeout=OPTUNA_TIMEOUT)
    print(f"best_params: {study.best_params}")
    LossHistory.clear_checkpoint()
    AccuracyHistory.clear_checkpoint()

    elapsed_time = time.time() - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    print(f"It took {hours} hours, {minutes} minutes, and {seconds:.2f} seconds to run.")