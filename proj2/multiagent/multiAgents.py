# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from game import Actions

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        x, y = successorGameState.getPacmanPosition()
        one_hops = set([(x+1, y), (x-1, y), (x, y-1), (x, y+1)])
        food = successorGameState.getFood()
        ghosts = set([ghost.getPosition() for ghost in successorGameState.getGhostStates()])
        successor_score = successorGameState.getScore()

        if not one_hops.intersection(ghosts) == set():
          successor_score -= 30

        if food[x][y]:
          successor_score += 100

        return successor_score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        def _getSubtreeAction(state, index, depth):
          #Reset index as necessary
          if index == state.getNumAgents():
              depth += 1
              index = 0
              #Base case
              if depth == self.depth:
                return (self.evaluationFunction(state), None)

          #Generate successor states; if there aren't any legal actions, return the current state
          node = {
            'type': 'min' if index > 0 else 'max',
            'children': {}
          }
          legal_actions = state.getLegalActions(index)
          if legal_actions == list():
            return (self.evaluationFunction(state), None)
          for action in legal_actions:
            node['children'][action] = _getSubtreeAction(state.generateSuccessor(index, action), index + 1, depth)

          #Retrieve appropriate action-score pair and return
          if node['type'] == 'min':
            min_pair = min(node['children'].iteritems(), key=lambda pair: pair[1][0])
            return (min_pair[1][0], min_pair[0])
          elif node['type'] == 'max':
            max_pair = max(node['children'].iteritems(), key=lambda pair: pair[1][0])
            return (max_pair[1][0], max_pair[0])


        action = _getSubtreeAction(gameState, 0, 0)
        print action
        return action[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        def _getSubtreeAction(state, index, depth, alpha, beta):
          #Reset index as necessary
          if index == state.getNumAgents():
              depth += 1
              index = 0
              #Base case; will always be returned from min node, so minimize beta
              if depth == self.depth:
                return (None, self.evaluationFunction(state))

          #Generate successor states; if there aren't any legal actions, return the current state
          node = {
            'type': 'min' if index > 0 else 'max',
            'best_move': (None, float('inf')) if index > 0 else (None, float('-inf'))
          }

          #If there aren't any legal moves, it's a leaf node
          legal_actions = state.getLegalActions(index)
          if legal_actions == list():
            return (None, self.evaluationFunction(state))

          #Branch subtree
          for next_action in legal_actions:
            move, score = _getSubtreeAction(state.generateSuccessor(index, next_action),
                                            index + 1, depth, alpha, beta)
            #modify alpha for max node
            if node['type'] == 'max':
              if score > alpha:
                alpha = score
              if score > node['best_move'][1]:
                node['best_move'] = (next_action, score)
              if beta < alpha:
                return node['best_move']

            #Modify beta for min node
            if node['type'] == 'min':
              if score < beta:
                beta = score
              if score < node['best_move'][1]:
                node['best_move'] = (next_action, score)
              if beta < alpha:
                return node['best_move']

          return node['best_move']

        #Root call
        action = _getSubtreeAction(gameState, 0, 0, float("-inf"), float("inf"))
        return action[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """


        def _getSubtreeAction(state, index, depth, alpha, beta):
          #Reset index as necessary
          if index == state.getNumAgents():
              depth += 1
              index = 0
              #Base case; will always be returned from min node, so minimize beta
              if depth == self.depth:
                return (None, self.evaluationFunction(state))

          node = {
            'type': 'min' if index > 0 else 'max'
          }
          if node['type'] == 'max':
            node['best_move'] = (None, float('-inf'))
          elif node['type'] == 'min':
            node['sum'] = 0

          #If there aren't any legal moves, it's a leaf node
          legal_actions = state.getLegalActions(index)
          if legal_actions == list():
            return (None, self.evaluationFunction(state))

          #Branch subtree
          for next_action in legal_actions:
            move, score = _getSubtreeAction(state.generateSuccessor(index, next_action),
                                            index + 1, depth, alpha, beta)
            #modify alpha for max node
            if node['type'] == 'max':
              if score > alpha:
                alpha = score
              if score > node['best_move'][1]:
                node['best_move'] = (next_action, score)
              if beta < alpha:
                return node['best_move']

            #Modify beta for min node
            if node['type'] == 'min':
              node['sum'] += float(score) / float(len(legal_actions))

          if node['type'] == 'max':
            return node['best_move']
          elif node['type'] == 'min':
            return (None, node['sum'])

          return node['best_move']

        #Root call
        action = _getSubtreeAction(gameState, 0, 0, float("-inf"), float("inf"))
        return action[0]
 
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    def _manhattan_distance(p1, p2):
      return abs(p2[1] - p1[1]) + abs(p2[0] - p1[0])

    score = currentGameState.getScore()
    food = currentGameState.getFood()
    pos = currentGameState.getPacmanPosition()

    #Sum the value of the food, divided by the square of the Manhattan distance from PacMan
    #Note: food is worth 10 points each
    sum = 0
    for x in range(food.width):
      for y in range(food.height):
        if food[x][y]:
          sum += 10 / _manhattan_distance(pos, (x, y))


    return sum + score



# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

