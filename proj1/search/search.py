# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    #Initializations
    visited = set()
    directions = None
    fringe = util.Stack()
    fringe.push((problem.getStartState(), []))

    #Iterate on fringe
    while not fringe.isEmpty():
        state = fringe.pop()

        #Check if the current state is a goal state
        if problem.isGoalState(state[0]):
            directions = state[1]
            break

        #Check if the state has been visited before, continue iteration
        elif state[0] in visited:
            continue

        #Otherwise, expand the node and say it was visited
        else:
            visited.add(state[0])
            for node in problem.getSuccessors(state[0]):
                if node[0] in visited:
                    continue
                dirs = list(state[1])
                dirs.append(node[1])
                fringe.push((node[0], dirs))

    #Munge and return the directions
    from game import Directions
    dirmap = {
        'South': Directions.SOUTH,
        'West': Directions.WEST,
        'North': Directions.NORTH,
        'East': Directions.EAST
    }
    return map(lambda val: dirmap[val], directions)
  

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    #Initializations
    visited = set()
    directions = None
    fringe = util.Queue()
    fringe.push((problem.getStartState(), []))

    #Iterate on fringe
    while not fringe.isEmpty():
        state = fringe.pop()

        #Check if the current state is a goal state
        if problem.isGoalState(state[0]):
            directions = state[1]
            break

        #Check if the state has been visited before, continue iteration
        elif state[0] in visited:
            continue

        #Otherwise, expand the node and say it was visited
        else:
            visited.add(state[0])
            for node in problem.getSuccessors(state[0]):
                if node[0] in visited:
                    continue
                dirs = list(state[1])
                dirs.append(node[1])
                fringe.push((node[0], dirs))

    #Munge and return the directions
    from game import Directions
    dirmap = {
        'South': Directions.SOUTH,
        'West': Directions.WEST,
        'North': Directions.NORTH,
        'East': Directions.EAST
    }
    print(type(directions))
    return map(lambda val: dirmap[val], directions)


def uniformCostSearch(problem):
    "Search the node of least total cost first. "

    #Initializations
    visited = {}
    directions = None
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), [], 0), 0)

    #Iterate on fringe
    while not fringe.isEmpty():
        state = fringe.pop()
        cost = state[2]

        #Check if the current state is a goal state
        if problem.isGoalState(state[0]):
            directions = state[1]
            break

        #Check if the state has been visited before;
        #if it has been and the new cost is higher, continue iteration
        elif state[0] in visited.keys():
            if cost >= visited[state[0]]:
                continue

        #Otherwise, expand the node and say it was visited
        visited[state[0]] = cost
        for node in problem.getSuccessors(state[0]):
            #Skip the node if the cost is higher than the current value
            # if node[0] in visited.keys():
            #     if cost + node[2] >= visited[node[0]]:
            #         continue
            dirs = list(state[1])
            dirs.append(node[1])
            fringe.push((node[0], dirs, cost + node[2]), cost + node[2])

    #Munge and return the directions
    from game import Directions
    dirmap = {
        'South': Directions.SOUTH,
        'West': Directions.WEST,
        'North': Directions.NORTH,
        'East': Directions.EAST
    }
    return map(lambda val: dirmap[val], directions)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    #Initializations
    visited = {}
    directions = None
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), [], 0), 0)

    #Iterate on fringe
    while not fringe.isEmpty():
        state = fringe.pop()
        cost = state[2]

        #Check if the current state is a goal state
        if problem.isGoalState(state[0]):
            directions = state[1]
            break

        #Check if the state has been visited before;
        #if it has been and the new cost is higher, continue iteration
        elif state[0] in visited.keys():
            if cost >= visited[state[0]]:
                continue

        #Otherwise, expand the node and say it was visited
        visited[state[0]] = cost
        for node in problem.getSuccessors(state[0]):
            #Skip the node if the cost is higher than the current value
            # if node[0] in visited.keys():
            #     if cost + node[2] + heuristic(state[0], problem) >= visited[node[0]]:
            #         continue
            dirs = list(state[1])
            dirs.append(node[1])
            fringe.push((node[0], dirs, cost + node[2]), cost + node[2] + heuristic(node[0], problem))

    #Munge and return the directions
    from game import Directions
    dirmap = {
        'South': Directions.SOUTH,
        'West': Directions.WEST,
        'North': Directions.NORTH,
        'East': Directions.EAST
    }
    return map(lambda val: dirmap[val], directions)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
