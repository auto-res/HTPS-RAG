import os
from datetime import datetime

HOME = os.environ['HOME']

PUCT_C = 1.0

if 'EXPERIMENT_DIR' not in globals():
    global EXPERIMENT_DIR
    now = datetime.now()
    EXPERIMENT_DIR = now.strftime("%Y%m%d_%H%M")

BASE_DIR = 'src/data'
# 実験用のデータディレクトリ
DATA_DIR = os.path.join(BASE_DIR, EXPERIMENT_DIR)

# データファイルのパス
VISUALIZED_DATA_PATH = os.path.join(DATA_DIR, 'visualized_data.json')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data.ndjson')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data.ndjson')
RESULT_DATA_PATH = os.path.join(DATA_DIR, 'result_data.ndjson')

MODEL_PATH = os.path.join(DATA_DIR, 'model.pt')
OPTIMIZER_PATH = os.path.join(DATA_DIR, 'optimizer.pt')
CRITIC_OPTIMIZER_PATH = os.path.join(DATA_DIR, 'critic_optimizer.pt')
CRITIC_LINEAR_PATH = os.path.join(DATA_DIR, 'critic_linear.pt')
# データディレクトリが存在しない場合は作成
os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_EPOCH = 1

EPOCH_SIZE = 4
BATCH_SIZE = 4
TRY_COUNT_LIMIT = 64

TRAIN_DATA_BATCH_SIZE = 32

WARMUP_STEPS = 100

EXPANSION_LIMIT = 1000

TEMPERATURE = 2.0

SOLVED_WEIGHT = 1.0
VALID_WEIGHT = 1.0
INVALID_WEIGHT = 0.0

RAG_LR = 0.001
ESTIMATOR_LR = 0.001

REPLAY_BUFFER_SIZE = 10000
REPLAY_BUFFER_SAMPLE = 1000

TIMEOUT = 170
NUM_SAMPLED_TACTICS = 10

CORPUS_FILENAME = 'corpus.jsonl'
INDEXED_CORPUS_FILENAME = 'indexed_corpus.pickle'

MINIF2F_BENCHMARK = "miniF2F_benchmark"
LEANDOJO_BENCHMARK = "leandojo_benchmark"
LEANDOJO_BENCHMARK_4 = "leandojo_benchmark_4"

OPTUNA_TIMEOUT = 60*60*24 - 120
