"""Proof search using best-first search.
"""
import os
import time
import torch
import numpy as np
from lean_dojo import (
    Pos,
    Dojo,
    Theorem,
    LeanGitRepo,
    TacticState,
    LeanError,
    TimeoutError,
    ProofFinished,
    ProofGivenUp,
    TacticResult,
    DojoInitError,
    DojoCrashError,
    DojoHardTimeoutError,
)
from loguru import logger
from dataclasses import dataclass
from typing import List, Optional, Tuple
from lean_dojo.constants import LEAN3_DEPS_DIR, LEAN4_DEPS_DIR

from app.common import zip_strict
from app.prover.hyper_tree import *
from app.constants import EXPANSION_LIMIT, TRY_COUNT_LIMIT, TEMPERATURE


class SearchResultStatus(Enum):
    """
    Enum class to represent the different statuses of a search result.

    - FAILED_EARLY: The search failed at an early stage.
    - SEARCH_TIMEOUT: The search was not completed due to timing out.
    - PROVED_BUT_TIMEOUT: A proof was found but not within the allotted time.
    - TOO_MUCH_EXPANSION: The search expanded beyond the allowable scope.
    - FOUND_A_PROOF: A valid proof was found.
    """
    FAILED_EARLY = "Failed early"
    SEARCH_TIMEOUT = "Search timeout"
    PROVED_BUT_TIMEOUT = "Proved but timeout"
    TOO_MUCH_EXPANSION = "Too much expansion"
    DOJO_CRASH_ERROR = "Dojo crash error"
    HTPS_CRASH_ERROR = "HTPS crash error"
    FOUND_A_PROOF = "Found a proof"

@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""

    theorem: Theorem
    status: Status
    search_result_status: SearchResultStatus
    proof: Optional[List[str]]
    hypergraph: HyperGraph

    # Some statistics during proof search.
    actor_time: float
    environment_time: float
    total_time: float
    num_total_nodes: int

    def to_dict(self):
        data_dict = {
            "theorem": {
                "repo": self.theorem.repo.name,
                "file_path": str(self.theorem.file_path),
                "full_name": self.theorem.full_name,
            },
            "status": self.status.value,
            "search_result_status": self.search_result_status.value,
            "proof": [] if not self.proof else self.proof,
            "actor_time": self.actor_time,
            "environment_time": self.environment_time,
            "total_time": self.total_time,
            "num_total_nodes": self.num_total_nodes
        }
        return data_dict


class HyperTreeProofSearchProver:
    def __init__(
        self,
        tac_gen,  # A given tactic generator.
        ver_est,  # A given verifiability estimator.
        timeout: int,
        num_sampled_tactics: int,
        temperature: float = TEMPERATURE,
        num_expansion: int = EXPANSION_LIMIT
    ) -> None:
        self.tac_gen = tac_gen
        self.ver_est = ver_est
        self.timeout = timeout
        self.temperature = temperature
        self.num_sampled_tactics = num_sampled_tactics
        self.num_expansion = num_expansion

        self.actor_time = 0.0
        self.environment_time = 0.0
        self.total_time = None

    def batch_search(self, repo: LeanGitRepo, theorems: List[Theorem], positions: List[Pos]) -> List[Optional[SearchResult]]:
        return [self.search(repo, theorem, pos) for theorem, pos in zip_strict(theorems, positions)]

    def batch_search_by_passn(
        self,
        repo: LeanGitRepo,
        theorems: List[Theorem],
        positions: List[Pos],
        try_count_limit: int = TRY_COUNT_LIMIT
    ) -> List[Optional[SearchResult]]:
        return [self.search_by_passn(repo, theorem, pos, try_count_limit) for theorem, pos in zip_strict(theorems, positions)]

    def search_by_passn(
        self,
        repo: LeanGitRepo,
        theorem: Theorem,
        position: Pos,
        try_count_limit: int = TRY_COUNT_LIMIT
    ) -> Optional[SearchResult]:
        result = None
        for _ in range(try_count_limit):
            result = self.search(repo, theorem, position)
            if result is None:
                break
            # If the result is the following, the result will not be changed.
            # Therefore we do not have to search the theorem anymore.
            if result.search_result_status in {SearchResultStatus.FOUND_A_PROOF, SearchResultStatus.DOJO_CRASH_ERROR, SearchResultStatus.FAILED_EARLY}:
                break
        return result

    def search(
        self, repo: LeanGitRepo, thm: Theorem, pos: Pos
    ) -> Optional[SearchResult]:
        logger.info(f"Proving {thm}")

        self.repo = repo
        self.theorem = thm
        self.position = pos
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.update_verifiability_call_count = 0

        try:
            with Dojo(thm, hard_timeout=60 + self.timeout) as (dojo, init_state):
                self.dojo = dojo
                self.root = InternalNode(
                    goal=init_state.goals[0]
                )
                self.hypergraph = HyperGraph(self.root)

                with torch.no_grad():
                    try:
                        search_result_status, proof = self._hyper_tree_proof_search(init_state)
                    except DojoCrashError:
                        logger.warning(f"Dojo crashed when proving {thm}")
                        search_result_status = SearchResultStatus.DOJO_CRASH_ERROR
                        if self.root.status == Status.PROVED:
                            logger.debug("self.root.status == Status.PROVED but dojo crashed. Replace the status with Status.FAILED")
                            self.root.status = Status.FAILED
                        pass
                    except HTPSCrashError:
                        logger.warning(f"HTPS crashed when proving {thm}")
                        search_result_status = SearchResultStatus.HTPS_CRASH_ERROR
                        if self.root.status == Status.PROVED:
                            logger.debug("self.root.status == Status.PROVED but htps crashed. Replace the status with Status.FAILED")
                            self.root.status = Status.FAILED
                        pass

                if search_result_status == SearchResultStatus.FOUND_A_PROOF:
                    # validate the proof is exactly a proof
                    current_state = init_state
                    for proof_step in proof:
                        try:
                            current_state = dojo.run_tac(current_state, proof_step)
                        except:
                            logger.warning(f"theorem: {thm.file_path}, {thm.full_name}")
                            logger.warning(f"whole proof: {proof}")
                            logger.warning(f"error deriving proof step: {proof_step}")
                            logger.warning(f"current_state: {current_state}")
                            raise HTPSCrashError(f"the theorem is considered proved, but the proof occurs an error, when proving {thm}.")

                    if not isinstance(current_state, ProofFinished):
                        raise HTPSCrashError(f"the theorem is considered proved, but the proof is not exactly a proof, when proving {thm}.")

                else:
                    proof = None

            result = SearchResult(
                theorem=thm,
                status=self.root.status,
                search_result_status=search_result_status,
                proof=proof,
                hypergraph=self.hypergraph,
                actor_time=self.actor_time,
                environment_time=self.environment_time,
                total_time=self.total_time,
                num_total_nodes=len(self.hypergraph)
            )
            logger.info(result.to_dict())
            return result

        except DojoInitError as ex:
            logger.warning(ex)
            return None

    def _hyper_tree_proof_search(self, init_state) -> Tuple[SearchResultStatus, Optional[List[str]]]:
        time_start = time.monotonic()

        while True:
            try:
                proof = self._step(init_state)
            except DojoHardTimeoutError:
                logger.debug(init_state)
                assert time.monotonic() - time_start >= self.timeout

            if len(self.hypergraph) > self.num_expansion:
                logger.info(f"It has exceeded the maximum number of attempts.")
                return SearchResultStatus.TOO_MUCH_EXPANSION, None

            self.total_time = time.monotonic() - time_start
            if self.total_time > self.timeout:
                if self.root.status == Status.PROVED:
                    logger.info("Found a proof but timed out.")
                logger.info("Search timed out.")
                return SearchResultStatus.SEARCH_TIMEOUT, proof

            if self.root.status == Status.FAILED:
                logger.info("Failed early!")
                return SearchResultStatus.FAILED_EARLY, None

            if proof is not None:
                logger.info("Found a proof!")
                return SearchResultStatus.FOUND_A_PROOF, proof

    # a cycle of proof search
    def _step(self, init_state: TacticState) -> Optional[List[str]]:
        logger.info(len(self.hypergraph))
        current_result: TacticResult = init_state
        history: list[tuple[TacticState, Optional[Edge]]] = []
        leads_cycle: bool = False
        # while we haven't reached an unexplored node, select a child node.
        while True:
            logger.info("step")
            logger.debug(f"root status: {self.root.status}")
            if isinstance(current_result, Union[LeanError, TimeoutError, ProofGivenUp]):
                break

            if self.root.status == Status.FAILED:
                logger.info("failed early")
                return None

            if isinstance(current_result, ProofFinished):
                self.root.status = Status.PROVED
                return [sas[1].tactic for sas in history]

            # assert current_state is TacticState
            goal = current_result.goals[0]
            key = goal.pp
            if not key in self.hypergraph:
                search_node = InternalNode(goal)
                self.hypergraph[key] = search_node
                break

            search_node = self.hypergraph[key]
            if not search_node.is_explored:
                break

            # assert search_node.is_explored
            selected_edge = self.select(search_node, self.temperature)
            if selected_edge is None:
                search_node.status = Status.FAILED

            if search_node.status == Status.FAILED:
                history.append((current_result, selected_edge))
                break

            state_before = current_result
            current_result = self.dojo.run_tac(state_before, selected_edge.tactic)

            # if the tactic leads to cycle, then we stop the search
            leads_cycle = self._leads_cycle(history, current_result)
            if leads_cycle:
                logger.info("leads cycle")
                selected_edge.update(0.0, invalid=True)
                history.append((state_before, selected_edge))
                break
            # apply the selected tactic
            history.append((state_before, selected_edge))


        if not isinstance(current_result, Union[LeanError, TimeoutError, ProofGivenUp])\
            and search_node.status != Status.FAILED\
            and not leads_cycle:
            edge = self.expand(search_node, current_result)
            if edge is not None:
                self.root.status = Status.PROVED
                return [sas[1].tactic for sas in history] + [edge.tactic]

        self.backpropagate(history, current_result, search_node.status)


    def select(self, node: InternalNode, temperature: float) -> Optional[Edge]:
        # assert node.is_explored
        for edge in node.out_edges:
            if edge.action_value == 1.0:
                return edge

        legal_edges = [edge for edge in node.out_edges if edge.action_value != 0.0]
        if len(legal_edges) == 0:
            return None

        PUCT_values = np.array([edge.puct(node.visit_count) for edge in legal_edges])

        if temperature == 0.0:
            return legal_edges[np.argmax(PUCT_values)]

        # substract max value to prevent overflowing
        PUCT_values -= np.max(PUCT_values)

        probabilities = np.exp(PUCT_values / temperature)
        probabilities /= np.sum(probabilities)

        # select an edge stochastically
        return np.random.choice(legal_edges, p=probabilities)

    def expand(self, node: InternalNode, state: TacticState) -> Optional[Edge]:
        # assert not node.is_explored
        suggestions = self._generate_tactics(node.goal.pp)

        out_edges = []
        for tactic, logprob in suggestions:
            state_after = self.dojo.run_tac(state, tactic)
            if isinstance(state_after, ProofFinished):
                edge = Edge(tactic=tactic, logprob=logprob, dsts=[ProofFinishedNode(state_after)])
                edge.update(1.0, proving=True)
                out_edges.append(edge)
                break

            if isinstance(state_after, Union[LeanError, TimeoutError, ProofGivenUp]):
                edge = Edge(tactic=tactic, logprob=logprob, dsts=[ErrorNode(state_after)])
                edge.update(0.0, invalid=True)
                # if update method is called, visit count is incrimented. Thus we revert.
                # edge.visit_count -= 1
            else:
                new_goal_len = len(state_after.goals) - len(state.goals) + 1
                dsts = [InternalNode(goal) for goal in state_after.goals[:new_goal_len]]
                edge = Edge(tactic=tactic, logprob=logprob, dsts=dsts)

            out_edges.append(edge)


        node.out_edges = out_edges
        if isinstance(state_after, ProofFinished):
            return edge

        if all(edge.action_value == 0.0 for edge in node.out_edges):
            node.status = Status.FAILED
        return None

    def backpropagate(self, history: list[tuple[TacticState, Optional[Edge]]], state_after: TacticResult, status: Status) -> None:
        #calculate score of state_after
        if isinstance(state_after, Union[LeanError, TimeoutError, ProofGivenUp]) or status == Status.FAILED:
            score = 0.0
        elif isinstance(state_after, ProofFinished):
            score = 1.0
        else:
            goal_pp = state_after.goals[0].pp
            score = self._estimate_verifiability(goal_pp)

        for state_before, edge in reversed(history):
            score = self._update_hypergraph(state_before, edge, state_after, status, score)
            state_after = state_before

    def _leads_cycle(self, history: list[tuple[TacticState, Edge]], state_after: TacticResult) -> bool:
        if not isinstance(state_after, TacticState):
            return False

        for state, _ in reversed(history):
            if state.goals[0].pp == state_after.goals[0].pp:
                if state.goals[0].pp in self.hypergraph:
                    node = self.hypergraph[state.goals[0].pp]
                    if node.status == Status.PROVED:
                        return False
                return True

    def _generate_tactics(self, ts: str) -> List[Tuple[str, float]]:
        t0 = time.monotonic()

        path = str(self.theorem.file_path)

        if self.theorem.repo != self.repo:
            if self.theorem.repo.uses_lean3:
                path = os.path.join(LEAN3_DEPS_DIR, self.theorem.repo.name, path)
            elif self.theorem.repo.is_lean:
                raise NotImplementedError
                path = os.path.join(LEAN4_DEPS_DIR, "lean4", path)
            else:
                path = os.path.join(LEAN4_DEPS_DIR, self.theorem.repo.name, path)

        suggestions = self.tac_gen.generate(
            state=ts,
            file_path=path,
            theorem_full_name=self.theorem.full_name,
            theorem_pos=self.position,
            num_samples=self.num_sampled_tactics,
        )

        self.actor_time += time.monotonic() - t0

        return suggestions

    def _update_hypergraph(self, state_before: TacticState, edge: Optional[Edge], state_after: TacticResult, status: Status, score: float) -> float:
        if edge is None:
            node = self.hypergraph[state_before.goals[0].pp]
            node.status = Status.FAILED
            return 0.0
        if isinstance(state_after, Union[LeanError, TimeoutError, ProofGivenUp]) or status == Status.FAILED:
            edge.update(0.0, invalid=True)
            return 0.0
        if self._partially_proved(state_before, state_after):
            edge.update(1.0, proving=True)
            return 1.0

        before_goal_len = len(state_before.goals)
        after_goal_len = len(state_after.goals)
        # assert after_goal_len >= before_goal_len
        new_goal_len = after_goal_len - before_goal_len + 1

        # we want to calculate the score of `state_before.goals[0]`
        # It is equals to the product of scores of `state_after.goals[:new_goal_len]`
        # Now we have `state_after.goals[0] = score` (= score of `state_after.goals[0]` = argument `score`)
        # In case i >= 1,
        # we let score of `state_after.goals[i]` be `_estimate_verifiability(state_after.goals[i].pp)`
        for goal in state_after.goals[1:new_goal_len]:
            score *= self._estimate_verifiability(goal.pp)

        edge.update(score)
        return score

    def _estimate_verifiability(self, goal_pp: str) -> float:
        if goal_pp in self.hypergraph:
            node = self.hypergraph[goal_pp]
            return node.verifiability

        return self.ver_est.estimate(goal_pp)


    def _partially_proved(self, state_before: TacticState, state_after: TacticResult) -> bool:
        if isinstance(state_after, ProofFinished):
            return True
        if len(state_before.goals) > len(state_after.goals):
            return True
        return False
