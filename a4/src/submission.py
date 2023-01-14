from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
# BEGIN_HIDE
# END_HIDE

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    # BEGIN_HIDE
    # END_HIDE

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # BEGIN_HIDE
    # END_HIDE
    return successorGameState.getScore()


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

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
    pass
    # ### START CODE HERE ###

    # isEnd() check
    # early exit on a win or loss state 
    if gameState.isWin() or gameState.isLose() or gameState.getLegalActions(0)==None:
      return Directions.STOP
    
    agents = gameState.getNumAgents()
    depthTemp = self.depth
    def recurse(state):
      activeAgent = self.index
      
      #pop back with the score once I reach full depth or game end
      if(state.isWin() or gameState.isLose() or (self.depth == 0 and self.index ==agents-1) or state.getLegalActions(activeAgent) == []):
        return(self.evaluationFunction(state),None)

      # determine the active agent
      if activeAgent == agents-1:
        self.depth -= 1
        self.index = 0
      else:
        self.index += 1

      actions = []
      legalActions = state.getLegalActions(activeAgent)
      for action in legalActions:
        actions.append((recurse(state.generateSuccessor(activeAgent,action))[0],action))
        #Choices = [(recurse(game.succ(state,action))[0], action) for action in game.actions(state)]
    
      # if pacman: Max
      if activeAgent == 0:
        opAction = max(actions)
      #if ghost: Min
      if activeAgent > 0:
        opAction = min(actions)      

      return opAction
    utility, action = recurse(gameState)
    self.depth = depthTemp
    self.index = 0
    #temp code to know the legal actions for this walk
    #pactions = gameState.getLegalActions(0)
    print('minimaxPolicy: => action {} with utility {}'.format(action, utility))
    return action
    ## ### END CODE HERE ###

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    pass
    # ### START CODE HERE ###
    if(gameState.isWin() or gameState.isLose() or gameState.getLegalActions()==[]):
      return Directions.STOP
    
    
    alpha = -1000
    beta = 2000

    agents = gameState.getNumAgents()
    tempDepth = self.depth

    def recurse(state):
      nonlocal alpha
      nonlocal beta
      activeAgent = self.index

      if(state.isWin() or gameState.isLose() or (self.depth == 0 and self.index ==agents-1) or state.getLegalActions() == []):
        return (self.evaluationFunction(state),None)
      
      if(activeAgent == agents-1):
        self.index = 0
        self.depth -= 1
      else:
        self.index += 1
      
      legalActions = state.getLegalActions(activeAgent) 
      actions = []
      #build the decision tree
      for action in legalActions:
        nxState = state.generateSuccessor(activeAgent,action)
        if activeAgent == 0 and self.evaluationFunction(nxState) > alpha:# and nxState.getScore() < beta:
          actions.append((recurse(nxState)[0],action))
        if activeAgent > 0 and self.evaluationFunction(nxState) < beta:# and nxState.getScore() > alpha:
          actions.append((recurse(nxState)[0],action))
        
      #TODO: it doesn't want to turn west for some reason
      if activeAgent == 0:
        opAction = max(actions)
        #this idea to fix being stuck in a corner is not a solution
        #for a in actions:
        #  if a[1] == action and ((a[0]-100) < opAction[0]):
        #    opAction = a
        alpha = opAction[0] if opAction[0] > alpha else alpha
      if activeAgent > 0:
        opAction = min(actions) 
        beta = opAction[0] if opAction[0] < beta else beta
      
      return opAction

    utility, action = recurse(gameState)

    #print("AlphaBeta Policy: action => {} with utility {}".format(action,utility))
    self.index = 0
    self.depth = tempDepth

    return action
      
    # ### END CODE HERE ###

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    pass
    # ### START CODE HERE ###

        # isEnd() check
    # early exit on a win or loss state 
    if gameState.isWin() or gameState.isLose() or gameState.getLegalActions(0)==None:
      return Directions.STOP
    
    agents = gameState.getNumAgents()
    depthTemp = self.depth
    def recurse(state): #regression loop
      activeAgent = self.index
      
      #pop back with the score once I reach full depth or game end
      if(state.isWin() or gameState.isLose() or (self.depth == 0 and self.index ==agents-1) or state.getLegalActions(activeAgent) == []):
        return(self.evaluationFunction(state),None)

      # determine the active agent
      if activeAgent == agents-1:
        self.depth -= 1
        self.index = 0
      else:
        self.index += 1

      actions = []
      legalActions = state.getLegalActions(activeAgent)
      for action in legalActions:
        actions.append((recurse(state.generateSuccessor(activeAgent,action))[0],action))
        #Choices = [(recurse(game.succ(state,action))[0], action) for action in game.actions(state)]
    
      # if pacman: Max
      if activeAgent == 0:
        opAction = max(actions)
      #if ghost: Min
      if activeAgent > 0:
        opActionIdx = random.randint(0,len(actions)-1)      
        opAction = actions[opActionIdx]

      return opAction
    utility, action = recurse(gameState)
    self.depth = depthTemp
    self.index = 0
    #temp code to know the legal actions for this walk
    #pactions = gameState.getLegalActions(0)
    #print('expectimaxPolicy: => action {} with utility {}'.format(action, utility))
    return action

    # ### END CODE HERE ###

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: <write something here so we know what you did>
  """
  pass
  # ### START CODE HERE ###
  curScore = currentGameState.getScore()
  #primary feature to add might be location to the capsules if im not in ghost mode and then location of ghosts if im in ghost eating mode
  # if there are no capsules search for food
  return curScore
  # ### END CODE HERE ###

# Abbreviation
better = betterEvaluationFunction
