## Retrieval-Augmented Generation
### Download latest leandojo_benchmark benchmark (including corpus.jsonl) 
```
python3 src/app/retrieval/download_corpus.py --data-path src/data
```
### test mathlib with RAG
```
python3 src/tests/test_mathlib_rag.py
```
### use retrirver for exploring any LeanGitRepo
LeanGitRepoでRAGつき学習を行う際には、コーパスの初期化が必要となります。
1. get_theorems_from_repo_url(url, commit, benchmark_name)でのベンチマークフォルダの作成
get_theorems_from_repo_urlでbenchmark_nameを指定すると、benchmark_nameの名前でベンチマークフォルダがtraceしたレポジトリなどをもとに作成されます(詳細はextract_data.pyの19行目以降)。
これをしない場合は上記のif文内の処理を実行したいコードに追加してください。
2. retrieverの初期化
ここではRetrievalAugmentedGeneratorでの初期化とsetup_model()での初期化の二つの方法があります。
 - RetrievalAugmentedGenerator
    minif2fで実験する場合、constats.pyにベンチマーク名を記入したのち、以下のようにコーパスへのパスを指定してRetrievalAugmentedGeneratorのcorpus_pathに渡してください。
    ~~~python
    from src.app.constants import DATA_DIR, CORPUS_FILENAME, MINIF2F_BENCHMARK
    ...
    corpus_path = f"{DATA_DIR}/{MINIF2F_BENCHMARK}/{CORPUS_FILENAME}"
    ~~~
 - setup_model()
    RAGを使いたい状況でsetup_model()を使用するときは、ベンチマークへのパスを渡してください。パスを渡さない場合RAGなしのgeneratorになります。
    ~~~python
    generator, estimator, prover, tokenizer = setup_model(benchmark=MINIF2F_BENCHMARK)
    ~~~

### Use original model on HuggingFace
retrieverやgeneratorで、Huggingfaceにある自分や所属組織のモデルを使用する場合、初回は`huggingface-cli login`を実行する必要があります。tokenの入力を要求されますので、以下のページでアクセストークンを発行して認証してください。
https://huggingface.co/settings/tokens