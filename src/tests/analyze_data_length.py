import matplotlib.pyplot as plt

def analyze_line_lengths(file_path):
    line_lengths = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 各行の長さをバイト単位で計測
            line_length = len(line.encode('utf-8'))
            line_lengths.append(line_length)
    return line_lengths

# ファイルパスを指定
file_path = 'src/tests/tests_data/fake_train_data.ndjson'

# 各行の長さを解析
line_lengths = analyze_line_lengths(file_path)

plt.figure(figsize=(10, 6))
plt.hist(line_lengths, bins=range(0, max(line_lengths) + 100, 100), edgecolor='black')
plt.title('Histogram of Line Lengths')
plt.xlabel('Length in Bytes')
plt.ylabel('Frequency')
plt.grid(True)
save_path = 'src/tests/tests_data/graph.png'  # 保存するパスとファイル名
plt.savefig(save_path)