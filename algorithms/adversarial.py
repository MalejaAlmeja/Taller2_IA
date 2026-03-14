from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """
        best_action = None
        best_value = float("-inf")

        for action in state.get_legal_actions(self.index):
            successor = state.generate_successor(self.index, action)
            value = self._minimax(successor, 1, self.depth - 1)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def _minimax(self, state: GameState, agent_index: int, depth: int) -> float:
        if state.is_win() or state.is_lose():
            return self.evaluation_function(state)

        if agent_index == 0 and depth == 0:
            return self.evaluation_function(state)

        num_agents = state.get_num_agents()
        legal_actions = state.get_legal_actions(agent_index)

        if not legal_actions:
            return self.evaluation_function(state)

        if agent_index == 0:
            best = float("-inf")
            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                best = max(best, self._minimax(successor, 1, depth - 1))
            return best
        else:
            next_agent = (agent_index + 1) % num_agents
            best = float("inf")
            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                best = min(best, self._minimax(successor, next_agent, depth))
            return best


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using alpha-beta pruning.

        Tips:
        - Same structure as MinimaxAgent, but with alpha-beta pruning.
        - Alpha: best value MAX can guarantee (initially -inf).
        - Beta: best value MIN can guarantee (initially +inf).
        - MAX node: prune when value > beta (strict inequality, do NOT prune on equality).
        - MIN node: prune when value < alpha (strict inequality, do NOT prune on equality).
        - Update alpha at MAX nodes: alpha = max(alpha, value).
        - Update beta at MIN nodes: beta = min(beta, value).
        - Pass alpha and beta through the recursive calls.
        """
        best_action = None
        best_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for action in state.get_legal_actions(self.index):
            successor = state.generate_successor(self.index, action)
            value = self._alphabeta(successor, 1, self.depth - 1, alpha, beta)

            if value > best_value:
                best_value = value
                best_action = action

            alpha = max(alpha, best_value)

        return best_action

    def _alphabeta(self, state, agent_index, depth, alpha, beta):
        if state.is_win() or state.is_lose():
            return self.evaluation_function(state)

        if agent_index == 0 and depth == 0:
            return self.evaluation_function(state)

        num_agents = state.get_num_agents()
        legal_actions = state.get_legal_actions(agent_index)

        if not legal_actions:
            return self.evaluation_function(state)

        if agent_index == 0:
            value = float("-inf")
            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                value = max(value, self._alphabeta(successor, 1, depth - 1, alpha, beta))
                alpha = max(alpha, value)

                if value > beta:
                    break
            return value
        else:
            next_agent = (agent_index + 1) % num_agents
            value = float("inf")
            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                value = min(value, self._alphabeta(successor, next_agent, depth, alpha, beta))
                beta = min(beta, value)

                if value < alpha:
                    break
            return value
    


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent with a mixed hunter model.

    Each hunter acts randomly with probability self.prob and greedily
    (worst-case / MIN) with probability 1 - self.prob.

    * When prob = 0:  behaves like Minimax (hunters always play optimally).
    * When prob = 1:  pure expectimax (hunters always play uniformly at random).
    * When 0 < prob < 1: weighted combination that correctly models the
      actual MixedHunterAgent used at game-play time.

    Chance node formula:
        value = (1 - p) * min(child_values) + p * mean(child_values)
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using expectimax with mixed hunter model.

        Tips:
        - Drone nodes are MAX (same as Minimax).
        - Hunter nodes are CHANCE with mixed model: the hunter acts greedily with
          probability (1 - self.prob) and uniformly at random with probability self.prob.
        - Mixed expected value = (1-p) * min(child_values) + p * mean(child_values).
        - When p=0 this reduces to Minimax; when p=1 it is pure uniform expectimax.
        - Do NOT prune in expectimax (unlike alpha-beta).
        - self.prob is set via the constructor argument prob.
        """
        best_action = None
        best_value = float("-inf")
        
        #Prueba cada acción del dron y evalúa con expectimax
        for action in state.get_legal_actions(self.index): 
            successor = state.generate_successor(self.index, action)
            value = self._expectimax(successor, 1, self.depth - 1)
            #Guarda la mejor acción 
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _expectimax(self, state: GameState, agent_index: int, depth: int) -> float:

        #Casos base (Iguales a los de arriba)
        if state.is_win() or state.is_lose():
            return self.evaluation_function(state)

        if agent_index == 0 and depth == 0:
            return self.evaluation_function(state)

        num_agents = state.get_num_agents()
        legal_actions = state.get_legal_actions(agent_index)

        if not legal_actions:
            return self.evaluation_function(state)

        # Nodo MAX: turno del dron (igual a Minimax)
        if agent_index == 0:
            best = float("-inf")
            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                best = max(best, self._expectimax(successor, 1, depth - 1))
            return best

        # Nodo de azar: turno del cazador con modelo mixto
        else:
            next_agent = (agent_index + 1) % num_agents #Si hay 2 cazadores hay que alternar
            child_values = []
            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                child_values.append(self._expectimax(successor, next_agent, depth))
            #Evalúa todos los movimientos del cazador, se miran todos porque se necesita para el promedio

            worst_case = min(child_values) # cazador juega perfecto
            average    = sum(child_values) / len(child_values) # cazador juega al azar

            # Formula de probabilidad
            return (1 - self.prob) * worst_case + self.prob * average
        

