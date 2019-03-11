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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
	
        #getting the list of foods in order to traverse it as a list
        listF = currentGameState.getFood().asList()
        #initializing the returning state as low as possible
        returning = -1000
        #initializing the manhattan distance between pacman and food
        manhattan = 0

        #if successor action is stop then we would just return one of the lowest value
        if action == Directions.STOP:
          return -1000

        # if the chosen newPos is where the ghost is at. Then we would definitely return the 
        #  lowest possible value
        for ghostState in newGhostStates:
          if ghostState.getPosition() == newPos and ghostState.scaredTimer == 0:
            return -1000
        
        # traversing through current available food list
        # choosing the nearest possbile food 
        # we are using negative numbers because the nearest food will get
        # the highest possible value
        for food in listF:
          manhattan = -1*(manhattanDistance(food,newPos))
          if(manhattan > returning):
            returning = manhattan
        return returning
        
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
        "*** YOUR CODE HERE ***"
        ## calling max on pacman itself which is at layer 0 
        return self.Max(gameState, 0)[1]
    

    def Max(self, gameState, level, agent = 0):
        #getting the possible legal actions
        lactions = gameState.getLegalActions(agent)
        #check whether I have an action, gamestate and level I am at
        if not lactions or gameState.isWin() or level >= self.depth:
            return self.evaluationFunction(gameState), Directions.STOP
        #Since I am getting the max value of the sub tree 
        #Initializing temporary variables as negative infinity and STOP action
        nCost = float('-inf')
        nAction = Directions.STOP

        # Giving thoughts on each legal actions
        for action in lactions:
            #create a successor on current agent with current legal action
            succ  = gameState.generateSuccessor(agent, action)
            #calling Min on new successor while maintaining the level but changing the agent.
            #whether agent is 0 or 1, adding 1 will make it a ghost agent.
            cost = self.Min(succ, level, agent + 1)[0]
            #if the new score with corresponding action from Min is better than old action then we will take that action 
            #Basically we are creating a successor to foresee the most suitable action.
            if cost > nCost:
                nCost = cost
                nAction = action
        return nCost, nAction
#Minimum layer 
    def Min(self, gameState, level, agent):
        # getting the legal actions possible at the current moment
        actions = gameState.getLegalActions(agent)
        # check we have legal actions,gameState and we are not exceeding possible depth
        if not actions or gameState.isLose() or level >= self.depth:
            return self.evaluationFunction(gameState), Directions.STOP
        #Since we are looking for the minimum possible value from nodes
        #we will initialize the temporary variables as infinity
        nCost = float('inf')
        nAction = Directions.STOP
        #Considering each legal actions
        for action in actions:
            #create a successor for that corresponding action
            succ = gameState.generateSuccessor(agent, action)
            #Initialize the possible score as 0
            cost = 0
            #If we already have considered all agents then we will go forward 
            #by 1 level
            if agent == gameState.getNumAgents() - 1:
                cost = self.Max(succ, level + 1)[0]
            #If we still have possible agents on this level then we create a successor 
            #Also we are calling min because agent!=0 is ghost.
            else:
                cost = self.Min(succ, level, agent + 1)[0]
            #If the foreseeing of the successor at current action is suitable
            #then we will take that action.
            if cost < nCost:
                nCost = cost
                nAction = action
                
        return nCost, nAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Initializing the alpha and beta parameters
        # calling max on pacman
        alpha = float('-inf')
        beta = float('inf')
        return self.getMaxValue(gameState, alpha, beta, 0)[1]

    def getMaxValue(self, gameState, alpha, beta, level, agent = 0):
        #legal actions available
        legals = gameState.getLegalActions(agent)
        #check whether legal actions are present and gamestate,current level
        if not legals or gameState.isWin() or level >= self.depth:
            return self.evaluationFunction(gameState), Directions.STOP
        #Initializing the maximizing value and giving it a base action
        succCost = float('-inf')
        succAct = Directions.STOP
        for action in legals:
            #generating successor at current legal action
            succ = gameState.generateSuccessor(agent, action)
            #getting the min value from that successor
            cost = self.getMinValue(succ, alpha, beta, level, agent + 1)[0]
            #check if that cost is bigger than so-far successor cost
            #swap the values
            if cost > succCost:
                succCost = cost
                succAct = action
            #if the successor cost is greater than beta then 
            if succCost > beta:
                return succCost, succAct
            #maximizing alpha at every successor 
            alpha = max(alpha, succCost)
        return succCost, succAct

    def getMinValue(self, gameState, alpha, beta, level, agent):
        # get legal actions 
        legals = gameState.getLegalActions(agent)
        # check termination states
        if not legals or gameState.isLose() or level >= self.depth:
            return self.evaluationFunction(gameState), Directions.STOP
        # initializing the minimizing temp values
        succCost = float('inf')
        succAct= Directions.STOP
        for action in legals:
            succ = gameState.generateSuccessor(agent, action)
            #initializing cost to some constant because I was having an error
            #when declaring it below
            cost = 0
            if agent == gameState.getNumAgents() - 1:
                #if we explored all the agent then go back to pacman and level up
                cost = self.getMaxValue(succ, alpha, beta, level + 1)[0]
            else:
                #if we haven't explored agents yet,then increase agent 
                cost = self.getMinValue(succ, alpha, beta, level, agent + 1)[0]
            #if the returned cost is less than previous one, then update
            if cost < succCost:
                succCost = cost
                succAct= action
            #if the new action value is less than alpha 
            if succCost < alpha:
                return succCost, succAct
            #always minimizing beta 
            beta = min(beta, succCost)
        return succCost, succAct

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    '''
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
    '''
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.getMaxValue(gameState, 0)[1]

    def getMaxValue(self, gameState, level, agent = 0):
        legals = gameState.getLegalActions(agent)
        if not legals or gameState.isWin() or level >= self.depth:
            return self.evaluationFunction(gameState), Directions.STOP
        succCost = float('-inf')
        succAct = Directions.STOP
        for action in legals:
            succ = gameState.generateSuccessor(agent, action)
            cost = self.expValue(succ, level, agent + 1)[0]
            if cost > succCost:
                succCost = cost
                succAct= action
        return succCost, succAct

    def expValue(self, gameState, level, agent):
        legals = gameState.getLegalActions(agent)
        if not legals or gameState.isLose() or level >= self.depth:
            return self.evaluationFunction(gameState), None
        succCosts = []
        for action in legals:
            succ = gameState.generateSuccessor(agent, action)
            cost = 0
            if agent == gameState.getNumAgents() - 1:
                cost = self.getMaxValue(succ, level + 1)[0]
            else:
                cost = self.expValue(succ, level, agent + 1)[0]
            succCosts.append(cost)
        #returning the exp value
        return sum(succCosts) / float(len(succCosts)), None

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: So for this part, It is almost similar as my first evaluation function.
      Except we are not basing on action where ghost and pacman are going.
      Pacman is only chasing the nearest foods around the map.
    """
    pacPos = list(currentGameState.getPacmanPosition())
    capPos = currentGameState.getCapsules()
    gh_state = currentGameState.getGhostStates()
    food_list   = currentGameState.getFood().asList()
    # the list of all manhattan distance between food and pacman
    hi = []
    for food in food_list:
        manhattan = manhattanDistance(food,pacPos)
        #-1 because the nearest will get the highest
        hi.append(-1*manhattan)
    if not hi:
        #if there's no food then just 0
        hi.append(0)
    #returning the nearest food + score
    #just to maximizing the returning value
    return max(hi) + currentGameState.getScore()
    "*** YOUR CODE HERE ***"
better = betterEvaluationFunction

