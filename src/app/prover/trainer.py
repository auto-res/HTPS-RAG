from src.app.generator.model import RLDataset
from torch.utils.data import DataLoader
from src.app.constants import (
    TRAIN_EPOCH,
    MODEL_PATH,
    OPTIMIZER_PATH,
    CRITIC_OPTIMIZER_PATH,
    CRITIC_LINEAR_PATH,
    TRAIN_DATA_BATCH_SIZE,
    REPLAY_BUFFER_SIZE,
    REPLAY_BUFFER_SAMPLE,
    TRAIN_DATA_PATH,
    SOLVED_WEIGHT,
    VALID_WEIGHT,
    INVALID_WEIGHT
    )

import random

from loguru import logger
import json

def _retrieve_from_replay_buffer(block_size=16384):
    with open(TRAIN_DATA_PATH, 'rb') as f:
        f.seek(0, 2)  # ファイルの末尾に移動
        filesize = f.tell()
        f.seek(max(filesize - block_size*REPLAY_BUFFER_SIZE, 0))  # ファイルの末尾から指定バイト数だけ戻る
        lines = f.readlines()  # バイト数分のデータを読み込む

    # 最後のREPLAY_BUFFER_SIZE行を取得
    replay_buffer = [json.loads(line.decode('utf-8')) for line in lines[-REPLAY_BUFFER_SIZE:]]

    random.shuffle(replay_buffer)
    return replay_buffer[:REPLAY_BUFFER_SAMPLE]

def train_model(generator, estimator, tokenizer):

    transformed_data = []

    # sampling_methodとして優先度付けサンプリングを実装するのであれば、学習時の誤差も必要になる
    for item in _retrieve_from_replay_buffer():
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

    assert len(transformed_data) > 0
    
    logger.debug(transformed_data[0])
    transformed_data = RLDataset(transformed_data, tokenizer)
    data = DataLoader(transformed_data, batch_size=TRAIN_DATA_BATCH_SIZE)

    total_gen_loss = 0
    total_est_loss = 0
    for _ in range(TRAIN_EPOCH):
        for batch in data:
            total_gen_loss += generator.train(batch)
            total_est_loss += estimator.train(batch)

            generator.save_model(OPTIMIZER_PATH)
            estimator.save_model(MODEL_PATH, CRITIC_OPTIMIZER_PATH, CRITIC_LINEAR_PATH)
            
    gen_loss = total_gen_loss/(TRAIN_DATA_BATCH_SIZE*TRAIN_EPOCH)
    est_loss = total_est_loss/(TRAIN_DATA_BATCH_SIZE*TRAIN_EPOCH)
    return gen_loss, est_loss