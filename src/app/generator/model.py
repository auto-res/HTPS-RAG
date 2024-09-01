import torch
import openai
from lean_dojo import Pos
from loguru import logger
from torchmetrics import Metric
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple, Union
import torch.nn.functional as F
import os
import pickle

from app.constants import (
    RAG_LR,
    ESTIMATOR_LR,
    WARMUP_STEPS,
    CORPUS_FILENAME,
    INDEXED_CORPUS_FILENAME
)
from app.common import (
    zip_strict,
    remove_marks,
    format_augmented_state,
    IndexedCorpus
)
from app.retrieval.model import PremiseRetriever

torch.set_float32_matmul_precision("medium")

class TopkAccuracy(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, batch_preds: List[List[str]], batch_gt: List[str]):
        assert len(batch_preds) == len(batch_gt)
        for preds, gt in zip(batch_preds, batch_gt):
            # This still doesn't account for short names vs. full names.
            gt = remove_marks(gt)
            preds = [remove_marks(p) for p in preds]
            self.correct += gt in preds[: self.k]
        self.total += len(batch_gt)

    def compute(self) -> float:
        return self.correct.float() / self.total


class TacticGenerator(ABC):
    """A tactic generator takes a state and generates multiple tactic candidates."""

    @abstractmethod
    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        raise NotImplementedError

class RetrievalAugmentedGenerator(TacticGenerator):
    def __init__(
        self,
        generator,
        tokenizer,
        num_beams: int,
        eval_num_retrieved: int,
        eval_num_cpus: int,
        eval_num_theorems: int,
        max_seq_len: int,
        optimizer=None,
        optimizer_path=None,
        lr: float = RAG_LR,
        warmup_steps: int = WARMUP_STEPS,
        warmup_level: int = 1,
        length_penalty: float = 0.0,
        retriever_name: Optional[str] = None,
        corpus_path: Optional[str] = None
    ) -> None:
        super().__init__()
        self.warmup_steps = warmup_steps
        self.num_beams = num_beams
        self.lr = lr
        self.warmup_level = warmup_level
        self.length_penalty = length_penalty
        self.eval_num_retrieved = eval_num_retrieved
        self.eval_num_cpus = eval_num_cpus
        self.eval_num_theorems = eval_num_theorems
        self.max_seq_len = max_seq_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if retriever_name is None:
            logger.info("Without retrieval")
            self.retriever = None
        else:
            logger.info(f"Loading the retriever")
            # GeneratorとRetrieverのlr, warmup_steps, max_seq_lenを一旦同じにして初期化。Retrieverも訓練する場合独自に引数を追加する必要あり。
            self.retriever = PremiseRetriever(retriever_name, lr, warmup_steps, max_seq_len).to(self.device)
            if corpus_path is not None:
                indexed_corpus_path = corpus_path[:-len(CORPUS_FILENAME)] + INDEXED_CORPUS_FILENAME
                if os.path.exists(indexed_corpus_path):
                    self.retriever.load_corpus(indexed_corpus_path)
                    logger.info(f"Indexed corpus loaded from {indexed_corpus_path}")
                else:
                    self.retriever.load_corpus(corpus_path)
                    self.retriever.reindex_corpus(batch_size=32)
                    pickle.dump(
                        IndexedCorpus(self.retriever.corpus, self.retriever.corpus_embeddings.cpu()),
                        open(indexed_corpus_path, "wb"),
                    )
                    logger.info(f"Indexed corpus saved to {indexed_corpus_path}")
            else:
                assert ValueError("corpus_path should be specified.")

        self.generator = generator.to(self.device)
        self.tokenizer = tokenizer

        if optimizer is None: #デフォルト
            self.optimizer = torch.optim.AdamW(self.generator.parameters(),lr=self.lr)
        else:
            self.optimizer = optimizer

        if optimizer_path is not None:
            self.optimizer.load_state_dict(torch.load(optimizer_path))

    def adjust_learning_rate(self):
        if self.warmup_level < self.warmup_steps:
            scale = float(self.warmup_level) / float(max(1, self.warmup_steps))
            lr_scaled = self.lr * scale  # 学習率をスケーリング
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_scaled
        self.warmup_level += 1

    ############
    # Training #
    ############

    def train(self, batch: Dict[str, Union[str, List[Dict[str, Any]], float]]):
        states = batch["state"]
        tactic = batch["tactic_ids"].squeeze(1).to(self.device) 
        visit_prob = batch["visit_prob"].to(self.device)
        learning_weight = batch["learning_weight"].to(self.device)

        # Get the model's output
        tokenized_states = self.tokenizer(states, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_len, return_attention_mask=True)
        input_ids = tokenized_states["input_ids"].to(self.device) 
        output = self.generator(input_ids=input_ids, labels=tactic).logits
        probs = F.softmax(output, dim=-1)
        tactic_prob = probs.gather(dim=-1, index=tactic.unsqueeze(-1)).squeeze(2)
        tactic_probs = torch.prod(tactic_prob, dim=-1)
        loss = torch.nn.CrossEntropyLoss(weight=learning_weight)(tactic_probs, visit_prob)

        self.adjust_learning_rate()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def save_model(self, optimizer_path):
        torch.save(self.optimizer.state_dict(), optimizer_path)  
              
    ##############
    # Prediction #
    ##############

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        return self.batch_generate(
            [state], [file_path], [theorem_full_name], [theorem_pos], num_samples
        )[0]

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        logger.debug(state)
        if self.retriever is not None:
            retrieved_premises, _ = self.retriever.retrieve(
                state,
                file_path,
                theorem_full_name,
                theorem_pos,
                self.eval_num_retrieved,
            )
            logger.debug(f'retrieved_premises: {retrieved_premises}')
            state = [
                format_augmented_state(s, premises, self.max_seq_len, p_drop=0.0)
                for s, premises in zip_strict(state, retrieved_premises)
            ]

        logger.debug(f"state: {state}")

        tokenized_state = self.tokenizer(
            state[0],
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        state_ids = tokenized_state.input_ids.to(self.device)
        state_mask = tokenized_state.attention_mask.to(self.device)

        # Generate tactic candidates using beam search.
        output = self.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            max_length=self.max_seq_len,
            num_beams=num_samples,
            length_penalty=self.length_penalty,
            do_sample=False,
            num_return_sequences=num_samples,
            early_stopping=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Return the output.
        raw_output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        raw_scores = output.sequences_scores.tolist()
        tactics_with_scores = []

        for i in range(len(state)):
            output_text = []
            output_score = []

            for j in range(i * num_samples, (i + 1) * num_samples):
                t = remove_marks(raw_output_text[j])
                if t not in output_text:
                    output_text.append(t)
                    output_score.append(raw_scores[j])

            tactics_with_scores.append(list(zip_strict(output_text, output_score)))

        return tactics_with_scores

class RLDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        state = item["state"]
        tactic = item["tactic"]

        tokenized_tactic = self.tokenizer(tactic, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        tactic_ids = tokenized_tactic.input_ids.squeeze(0)

        if item["parent_visit_count"] > 0:
            visit_prob = item["visit_count"] / item["parent_visit_count"]
        else:
            visit_prob = 1.0

        learning_weight = item["learning_weight"]
        verifiability = item["verifiability"]

        return {
            "state": state,
            "tactic_ids": tactic_ids,
            "visit_prob": visit_prob,
            "learning_weight": learning_weight,
            "verifiability": verifiability
        }

class GPT4TacticGenerator(TacticGenerator):
    def __init__(
        self,
        organization: str,
        api_key: str,
        model: str = "gpt-4",
        max_tokens: int = 1024,
        num_retries: int = 3,
        threshold: float = 0.9,
    ):
        super().__init__()
        openai.organization = organization
        openai.api_key = api_key
        self.model = model
        self.default_prompt = "You are an expert in Lean3 theorem proofs. We are trying to solve the Lean3 theorem 'THEOREM_FULL_NAME' from the mathlib file 'FILE_PATH'. The current tactic state is: 'TACTIC_STATE'. Suggest exactly NUM_SAMPLES unique tactics to progress in solving 'THEOREM_FULL_NAME', along with their confidence levels as a float between 0 and 1. Rank them in order of effectiveness. Present the tactics and their confidence levels as comma-separated tuples in this format: #(tactic_{1}, confidence_{1})#, #(tactic_{2}, confidence_{2})#, ..., #(tactic_{NUM_SAMPLES}, confidence_{NUM_SAMPLES})#."
        self.max_tokens = max_tokens
        self.num_retries = num_retries
        self.threshold = threshold

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        prompt = (
            self.default_prompt.replace("TACTIC_STATE", state)
            .replace("FILE_PATH", file_path)
            .replace("THEOREM_FULL_NAME", theorem_full_name)
            .replace("NUM_SAMPLES", str(int(num_samples / self.threshold)))
        )
        logger.info(prompt)

        for _ in range(self.num_retries):
            response = None
            # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    # temperature=0,
                    max_tokens=self.max_tokens,
                    # stop="E:" #
                )
            except openai.error.APIError as e:
                # Handle API error here, e.g. retry or log
                logger.info(f"OpenAI API returned an API Error: {e}")
                continue
            except openai.error.APIConnectionError as e:
                # Handle connection error here
                logger.info(f"Failed to connect to OpenAI API: {e}")
                continue
            except openai.error.RateLimitError as e:
                # Handle rate limit error (we recommend using exponential backoff)
                logger.info(f"OpenAI API request exceeded rate limit: {e}")
                continue
            except Exception as e:
                logger.info(e)
                continue

            if response is None:
                continue

            logger.info(f"GPT-4 response: {response}")
            output = response["choices"][0]["message"]["content"]
            indices = []

            for i, c in enumerate(output):
                if c == "#":
                    indices.append(i)

            tactics_with_scores = []

            for i in range(1, len(indices), 2):
                tactic_and_confidence = output[indices[i - 1] + 1 : indices[i]].strip()

                try:
                    while tactic_and_confidence[0] == "(":
                        tactic_and_confidence = tactic_and_confidence[1:]

                    if tactic_and_confidence[-1] == ")":
                        tactic_and_confidence = tactic_and_confidence[:-1]

                    split_index = tactic_and_confidence.rindex(",")
                    tactic = tactic_and_confidence[:split_index].strip()
                    confidence = float(tactic_and_confidence[split_index + 1 :].strip())
                except Exception as e:
                    logger.info(e)
                    logger.info(
                        f"{self.model} output {output[indices[i-1]+1:indices[i]]} was not formatted correctly and could not be parsed."
                    )
                    continue

                tactics_with_scores.append((tactic, confidence))

            if len(tactics_with_scores) < int(self.threshold * num_samples):
                continue

            tactics_with_scores = sorted(
                tactics_with_scores, key=lambda x: x[1], reverse=True
            )[: min(num_samples, len(tactics_with_scores))]
            logger.debug(f"GPT-4 tactics: {tactics_with_scores}")
            logger.debug(
                f"GPT-4 tactic count requested: {num_samples} / {self.threshold} = {int(num_samples / self.threshold)}"
            )
            logger.debug(
                f"GPT-4 tactic count received and parsed: {len(tactics_with_scores)}"
            )
            return tactics_with_scores

        raise ValueError("GPT-4 outputs are unparsable.")

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        return [
            self.generate(s, f, t, p, num_samples)
            for s, f, t, p in zip_strict(
                state, file_path, theorem_full_name, theorem_pos
            )
        ]


class VerifiabilityEstimator(ABC):
    """A verifiability estimator calculates the possibility to prove a given theorem."""
    
    @abstractmethod
    def estimate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str
    ) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def batch_estimate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str]
    ) -> List[float]:
        raise NotImplementedError

class HTPSVerifiabilityEstimator(VerifiabilityEstimator): 
    def __init__(
        self,
        base_generator,
        tokenizer,
        max_seq_len: int,
        warmup_steps: int = WARMUP_STEPS,
        lr: float = ESTIMATOR_LR,
        warmup_level: int = 1,
        critic_optimizer=None,
        linear_layers=None,
        critic_optimizer_path=None,
        critic_linear_path=None,
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_generator = base_generator.to(self.device)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.warmup_level = warmup_level

        if linear_layers is None:
            self.linear_layers = torch.nn.Sequential(
                torch.nn.Linear(self.base_generator.config.hidden_size, self.base_generator.config.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.base_generator.config.hidden_size, 1),
                torch.nn.Sigmoid()
            ).to(self.device)
        else:
            self.linear_layers = linear_layers.to(self.device)
        
        if critic_linear_path is not None:
            self.linear_layers.load_state_dict(torch.load(critic_linear_path))
        
        if critic_optimizer is None:
            self.critic_optimizer = torch.optim.AdamW(self.base_generator.parameters(),lr=self.lr)
        else:
            self.critic_optimizer = critic_optimizer
    
        if critic_optimizer_path is not None:
            self.critic_optimizer.load_state_dict(torch.load(critic_optimizer_path))

    def estimate(self, state: str) -> float:	
        return self.batch_estimate([state])[0] #<- 評価はstateの最後の文字で

    def batch_estimate(self, states: List[str]) -> List[float]:
        tokenized_states = self.tokenizer(
            states,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,  # または必要な最大長に設定
            return_tensors="pt"
        )  
        input_ids = tokenized_states.input_ids.to(self.device)
        attention_mask = tokenized_states.attention_mask.to(self.device)

        encoder_outputs = self.base_generator.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        final_output = self.linear_layers(encoder_outputs)
        return final_output[:, 0, -1].tolist() #<- 評価はstateの最後の文字で

    def adjust_learning_rate(self):
        if self.warmup_level < self.warmup_steps:
            scale = float(self.warmup_level) / float(max(1, self.warmup_steps))
            lr_scaled = self.lr * scale  # 学習率をスケーリング
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = lr_scaled
        self.warmup_level += 1

    def train(self, batch: Dict[str, Union[str, List[Dict[str, Any]], float]]):
        states = batch["state"]
        verifiabilities = batch["verifiability"].clone().detach().to(self.device)  # clone().detach()を使用してデバイスに移動
        estimates = torch.tensor(self.batch_estimate(states), requires_grad=True).to(self.device)
        loss = torch.mean(torch.square(verifiabilities - estimates))

        self.adjust_learning_rate()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        return loss.item()

    def save_model(self, model_path, critic_optimizer_path, critic_linear_path):
        torch.save(self.base_generator.state_dict(), model_path)
        torch.save(self.critic_optimizer.state_dict(), critic_optimizer_path)
        torch.save(self.linear_layers.state_dict(), critic_linear_path)