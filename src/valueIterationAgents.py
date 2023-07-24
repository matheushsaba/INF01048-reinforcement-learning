# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here
        for _ in range(self.iterations):
            stateToQValues = util.Counter()                                     # Dicionário (estado, valor Q)            
            allStates = mdp.getStates()                                         # Pega todos os estados possíveis

            for state in allStates:
                isTerminalState = mdp.isTerminal(state)
                possibleActions = self.mdp.getPossibleActions(state)

                if(isTerminalState or len(possibleActions) == 0):               # Se for estado terminal ou não há ações possíveis, pula para o próximo estado
                    continue
                else:                                                           # Senão, atualiza o valor Q de cada ação possível
                    possibleActions = mdp.getPossibleActions(state)
                    actionToQValueDictionary = util.Counter()                   # Dicionário (ação, valor Q)

                    for possibleAction in possibleActions:                      # Itera sobre as possíveis ações, calculando o valor Q de cada uma
                        qValue = self.getQValue(state, possibleAction)
                        actionToQValueDictionary[possibleAction] = qValue       # Adiciona os valores de Q(s,a) no dicionário

                    bestAction = actionToQValueDictionary.argMax()              # Pega a melhor ação
                    bestActionQValue = actionToQValueDictionary[bestAction]
                    stateToQValues[state] = bestActionQValue

            self.values = stateToQValues                                        # Atualiza os valores de V(s) com os valores de Q(s,a) calculados

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        nextStatesAndProbabilities = self.mdp.getTransitionStatesAndProbs(state, action)            # Lista de tuplas (próximo estado, probabilidade)
        accumulatedQValue = 0

        for nextStateAndProbability in nextStatesAndProbabilities:                                  # Itera sobre as tuplas (próximo estado, probabilidade)
            nextState = nextStateAndProbability[0]
            probability = nextStateAndProbability[1]
            reward = self.mdp.getReward(state, action, nextState)
            accumulatedQValue += probability * (reward + self.discount * self.values[nextState])    # Qk+1(s,a) = R(s,a) + gama * somatório[P(s'|s,a) * Vk(s')]

        return accumulatedQValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        possibleActions = self.mdp.getPossibleActions(state)

        if self.mdp.isTerminal(state) or len(possibleActions) == 0:             # Se estiver no estado terminal ou não há ações possíveis, retorna None
            return None
        else:                                                                   # Senão calcula a melhor ação possível de um dado estado
            actionToQValueDictionary = util.Counter()                           # Dicionário (ação, valor Q)
            for possibleAction in possibleActions:                              # Itera sobre as possíveis ações, calculando o valor Q de cada uma
                action = possibleAction
                qValue = self.getQValue(state, action)
                actionToQValueDictionary[action] = qValue
            
            bestAction = actionToQValueDictionary.argMax()                      # Pega a melhor ação

            return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
