# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        evaluation = 0

        if newFood[newPos[0]][newPos[1]] == True:
            evaluation += 1000

        count = 0
        
        
        for i in newFood.asList():
            manDistance = abs(newPos[0] - i[0]) + abs(newPos[1] - i[1])
            if manDistance <= 4:
                count += 1
        

        evaluation += 1.2 * count     
        
        
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            manDistance = abs(newPos[0] - ghostPos[0]) + abs(newPos[1] - ghostPos[1])
            if newScaredTimes[0] == 0:
                if manDistance == 0:
                    evaluation -= 5000
                elif manDistance <= 8:
                    evaluation -= 50 * (1/manDistance)

        evaluation += successorGameState.getScore()

        return evaluation 

def scoreEvaluationFunction(currentGameState: GameState):
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

    def minimax_value(self, state: GameState, depth, index):
        if state.isLose() or state.isWin():
            return self.evaluationFunction(state)

        if index > state.getNumAgents() - 1:
            if depth == 0:
                return self.evaluationFunction(state)
            return self.minimax_value(state, depth - 1, 0)

        min_value = 99999
        max_value = -99999

        if index == 0:
            for action in state.getLegalActions(index):
                newstate = state.generateSuccessor(index, action)
                max_value = max(max_value, self.minimax_value(newstate, depth, index + 1))
            return max_value
        else: 
            for action in state.getLegalActions(index):
                newstate = state.generateSuccessor(index, action)
                min_value = min(min_value, self.minimax_value(newstate, depth, index + 1))
            return min_value

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        result_action = None
        minmax_value = -99999
        for action in gameState.getLegalActions(0):
            newstate = gameState.generateSuccessor(0, action)
            if newstate.isWin() or newstate.isLose():
                value = self.evaluationFunction(newstate)
                if value > minmax_value:
                    minmax_value = value 
                    result_action = action
            else:  
                value = self.minimax_value(newstate, self.depth - 1, 1)
                if value > minmax_value:
                    minmax_value = value 
                    result_action = action
        
        return result_action
    
    
    


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def helper(self, state: GameState, index, depth, alpha, beta):
        if depth == self.depth:
            return self.evaluationFunction(state)
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if index == 0:
            return self.maxvalue(state, 0, depth, alpha, beta)
        else: 
            if index > state.getNumAgents() - 1:
                return self.helper(state, 0, depth + 1, alpha, beta)
            return self.minvalue(state, index, depth, alpha, beta)
    

    def maxvalue(self, state: GameState, index, depth, alpha, beta):
        max_value = -99999
        for action in state.getLegalActions(index):
            newstate = state.generateSuccessor(index, action)
            max_value = max(max_value, self.helper(newstate, index + 1, depth, alpha, beta))
            if max_value > beta:
                return max_value
            alpha = max(alpha, max_value)
        return max_value

    def minvalue(self, state: GameState, index, depth, alpha, beta):
        min_value = 99999
        for action in state.getLegalActions(index):
            newstate = state.generateSuccessor(index, action)
            min_value = min(min_value, self.helper(newstate, index + 1, depth, alpha, beta))
            if min_value < alpha:
                return min_value
            beta = min(beta, min_value)
        return min_value

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        result_action = None
        minmax_value = -99999
        alpha = -99999
        beta = 99999

        for action in gameState.getLegalActions(0):
            newstate = gameState.generateSuccessor(0, action)
            value = self.helper(newstate, 1, 0, alpha, beta)
            if value > minmax_value:
                minmax_value = value 
                result_action = action
                alpha = value
        
        return result_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def helper(self, state: GameState, index, depth):
        if depth == self.depth:
            return self.evaluationFunction(state)
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if index == 0:
            return self.maxvalue(state, 0, depth)
        else: 
            if index > state.getNumAgents() - 1:
                return self.helper(state, 0, depth + 1)
            return self.randomvalue(state, index, depth)
    

    def maxvalue(self, state: GameState, index, depth):
        max_value = -99999
        for action in state.getLegalActions(index):
            newstate = state.generateSuccessor(index, action)
            max_value = max(max_value, self.helper(newstate, index + 1, depth))
        return max_value

    def randomvalue(self, state: GameState, index, depth):
        random_value = []
        for action in state.getLegalActions(index):
            newstate = state.generateSuccessor(index, action)
            random_value = random_value + [self.helper(newstate, index + 1, depth)]
        
        sum = 0
        for i in random_value:
            sum += i
        return sum / len(random_value)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        result_action = None
        minmax_value = -99999

        for action in gameState.getLegalActions(0):
            newstate = gameState.generateSuccessor(0, action)
            value = self.helper(newstate, 1, 0)
            if value > minmax_value:
                minmax_value = value 
                result_action = action
        
        return result_action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    if the gamestate is win, then we return a maximized value for the evaluation function
    if the gamestate is lose, then we return a minimized value for the evaluation function

    Also, we don't want the pacman to be super close to ghost, so we deduct some evaluation point 
    if the manhattan distance between Pacman and ghost is closer than 8.

    We also want the pacman to move to the place with more food, so we add some evaluation point
    for place with more food.

    """
    "*** YOUR CODE HERE ***"
    evaluation = 0

    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    if currentGameState.isWin():
        return 999999
    
    if currentGameState.isLose():
        return -999999
    
    for ghost in ghostStates:
            ghostPos = ghost.getPosition()
            manDistance = abs(Pos[0] - ghostPos[0]) + abs(Pos[1] - ghostPos[1])
            if scaredTimes[0] == 0:
                if manDistance <= 8:
                    evaluation -= 50 * (1/manDistance)
    
    count = 0
    for i in Food.asList():
            manDistance = abs(Pos[0] - i[0]) + abs(Pos[1] - i[1])
            if manDistance <= 4:
                count += 1
        

    evaluation += 2 * count   

    evaluation += currentGameState.getScore()
    
    return evaluation

# Abbreviation
better = betterEvaluationFunction
