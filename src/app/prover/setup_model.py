from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.generator.model import RetrievalAugmentedGenerator, HTPSVerifiabilityEstimator
from app.prover.proof_search import HyperTreeProofSearchProver
from app.constants import TIMEOUT, NUM_SAMPLED_TACTICS, CORPUS_FILENAME, DATA_DIR
import torch
from pathlib import Path
from typing import Union

def setup_model(
        model_path="kaiyuy/leandojo-lean3-tacgen-byt5-small", 
        tokenizer_path="kaiyuy/leandojo-lean3-tacgen-byt5-small", 
        retriever_name="kaiyuy/leandojo-lean3-retriever-byt5-small", 
        benchmark:str=None, 
        saved_model_path=None, 
        optimizer_path=None, 
        critic_linear_path=None, 
        critic_optimizer_path=None,
        generator_config:dict=None,
        estimator_config:dict=None,
        prover_config:dict=None,
        ):

    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    base_generator = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    if saved_model_path is not None:
        base_generator.load_state_dict(torch.load(saved_model_path))
    base_generator.eval()

    corpus_path = f"{DATA_DIR}/{benchmark}/{CORPUS_FILENAME}"

    if generator_config is None:
        generator_config = {
            'num_beams': 1,
            'eval_num_retrieved': 1,
        }

    if prover_config is None:
        prover_config = {
            'num_sampled_tactics': NUM_SAMPLED_TACTICS
        }

    if estimator_config is None:
        estimator_config = {}

    generator = RetrievalAugmentedGenerator(
        base_generator,
        tokenizer,
        eval_num_cpus=1,
        eval_num_theorems=1,
        max_seq_len=512,
        optimizer_path=optimizer_path,
        retriever_name=retriever_name,
        corpus_path=corpus_path,
        **generator_config
        )
    
    estimator = HTPSVerifiabilityEstimator(
        base_generator,
        tokenizer,
        max_seq_len=512,
        critic_linear_path=critic_linear_path,
        critic_optimizer_path=critic_optimizer_path,
        **estimator_config
        )
    
    prover = HyperTreeProofSearchProver(
        generator, 
        estimator,
        timeout=TIMEOUT,
        **prover_config
    )

    return generator, estimator, prover, tokenizer