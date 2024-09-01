# HyperTree Proof Search with Retrieval Augmented Tactic Generator

## Hyper Tree Proof Search (HTPS): An Overview
Hyper Tree Proof Search (HTPS) is an AI model designed for mathematical theorem proving. It combines advanced machine learning techniques with sophisticated search algorithms to verify and prove mathematical theorems.

### Tree Search Algorithm
At the heart of HTPS is a tree search algorithm, which efficiently explores large decision spaces. This algorithm balances the exploration of new, potentially beneficial paths with the exploitation of paths already known to be effective. In the context of theorem proving, this allows HTPS to navigate through complex mathematical proofs systematically.

### Reinforcement Learning
HTPS utilizes reinforcement learning, a type of machine learning where the system learns optimal strategies through trial and error. This approach allows HTPS to improve its theorem-proving capabilities over time by adapting its strategies based on the outcomes of its decisions.

### Key Elements

**State Representation:** In HTPS, the 'state' refers to the current stage of a theorem being verified. This representation allows the system to track progress and make decisions based on the current proof status.

**Policy Model:** HTPS employs a policy model, potentially driven by a Large Language Model (LLM), to suggest the best tactic or the minimum unit of proof in theorem verification. This model can be thought of as performing text generation tasks, proposing the next logical step in the proof process.

**Critic Model:** The critic model in HTPS assesses the verifiability of the theorem under consideration. It acts as a text classification model, determining whether the policy model can provide a valid proof for the current state of the theorem.

Both the policy model and critic model in HTPS use the language model from [ReProver](https://github.com/lean-dojo/ReProver) as the target model for training.

## How to setup
1. Move to the docker directory
```bash
cd docker
```
2. Build the docker image and run the container
```bash
./build.sh
```
3. First, you need to install the LeanDojo in the container
```bash
cd ..
pip install -e src/app/LeanDojo/
```

## How to evaluate miniF2F
```bash
python src/app/gym/evaluate_minif2f.py
```

## Reference

Yang, K., Swope, A., Gu, A., Chalamala, R., Song, P., Yu, S., Godil, S., Prenger, R., & Anandkumar, A. (2023). LeanDojo: Theorem Proving with Retrieval-Augmented Language Models. In Proceedings of the Neural Information Processing Systems Conference (NeurIPS 2023), Datasets and Benchmarks Track, Oral presentation.
