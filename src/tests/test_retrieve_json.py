from app.constants import REPLAY_BUFFER_SIZE
import json

with open('src/tests/tests_data/fake_train_data.ndjson', 'rb') as f:
    f.seek(0, 2)  # ファイルの末尾に移動
    filesize = f.tell()
    f.seek(max(filesize - 16384*REPLAY_BUFFER_SIZE, 0))  # ファイルの末尾から指定バイト数だけ戻る
    lines = f.readlines()  # バイト数分のデータを読み込む

# 最後のREPLAY_BUFFER_SIZE行を取得
replay_buffer = [json.loads(line.decode('utf-8')) for line in lines[-REPLAY_BUFFER_SIZE:]]

print(replay_buffer[:5])