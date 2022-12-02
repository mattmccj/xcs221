from typing import Callable, List, Set

import shell
import util
import wordsegUtil

############################################################
# Problem 1b: Solve the segmentation problem under a unigram model


class SegmentationProblem(util.SearchProblem):
    def __init__(self, query: str, unigramCost: Callable[[str], float]):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        pass
        # ### START CODE HERE ###
        #this returns the state
        return 0 #start at the beginning of the string
        # ### END CODE HERE ###

    def isEnd(self, state) -> bool:
        pass
        # ### START CODE HERE ###
        #Have we reached the end of the string? T/F
        #what is the state? the current location in the string?
        if state >= len(self.query)-1: return True

        return False
        # ### END CODE HERE ###

    def succAndCost(self, state):
        pass
        # ### START CODE HERE ###
        #This is the recursive call that dials in the cost per action
        #should return action, newstate, and cost
             #(Action, state chge, cost)?
        result = []
        #end = len(self.query)
        end = state+7 if state+7 < len(self.query) else len(self.query)
        for i in range(state,end):
            word = self.query[state:i+1]
            result.append((word, i+1, self.unigramCost(word)+end-i))
        return result
        #curWord = self.query[state[0]:state[1]]
        #cost = self.unigramCost(curWord)
        #
        #wrdNxtState = [state[1]+1,state[1]+2]
        #
        #ltrNxtState = [state[0],state[1]+1] 
        #
        ##i have 2 clear actions, and a clear function that calculates cost
        #return  [(" ", wrdNxtState, cost), (self.query[state[1]], ltrNxtState, 0)]
        #return action, nxtState, cost
        # ### END CODE HERE ###


def segmentWords(query: str, unigramCost: Callable[[str], float]) -> str:
    if len(query) == 0:
        return ""

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # ### START CODE HERE ###
    #I think I need to build out the state object here it likely is a triple object tuple that contains the string, words, cost or something similar 
    
    output = ""
    state = 0
    #I need to wrappup the words into a string again at the end
    for i in range(len(ucs.actions)):
        output = output + ucs.actions[i]
        if i < len(ucs.actions)-1:
            output = output + " "
    
    return output
    # ### END CODE HERE ###


############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost


class VowelInsertionProblem(util.SearchProblem):
    def __init__(
        self,
        queryWords: List[str],
        bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]],
    ):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        pass
        # ### START CODE HERE ###
        return 0, wordsegUtil.SENTENCE_BEGIN
        # ### END CODE HERE ###

    def isEnd(self, state) -> bool:
        pass
        # ### START CODE HERE ###
        #if state has picked up all the words we are done
        if state[0] >= len(self.queryWords): return True

        return False
        # ### END CODE HERE ###

    def succAndCost(self, state):
        pass
        # ### START CODE HERE ###
            #Pick up the right 2 words
        result = []
        
        #TODO ???Need 1 wrd condition here???
        lwrd = state[1] #self.queryWords[state-1] if state != 0 else wordsegUtil.SENTENCE_BEGIN
        rwrd = self.queryWords[state[0]] 

        #lfills = self.possibleFills(lwrd)
        #if len(lfills) == 0: lfills= [lwrd]
        rfills = self.possibleFills(rwrd)
        if len(rfills) == 0: rfills= [rwrd]

        #for lfill in lfills:
        for rfill in rfills:
                result.append((rfill,(state[0]+1,rfill),self.bigramCost(lwrd,rfill)))
        return result
        #run them both through their possibleFills
        #zip the Fills together as states to move forward
        # ### END CODE HERE ###


def insertVowels(
    queryWords: List[str],
    bigramCost: Callable[[str, str], float],
    possibleFills: Callable[[str], Set[str]],
) -> str:
    pass
    # ### START CODE HERE ###
    if len(queryWords) == 0: return ""

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords,bigramCost,possibleFills))

    output = ""
    for action in ucs.actions:
        output = output + action
        if action != ucs.actions[-1]:
            output = output + " "
    return output
    # ### END CODE HERE ###


############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem


class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(
        self,
        query: str,
        bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]],
    ):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        pass
        # ### START CODE HERE ###
        return 0, wordsegUtil.SENTENCE_BEGIN
        # ### END CODE HERE ###

    def isEnd(self, state) -> bool:
        pass
        # ### START CODE HERE ###
        if state[0] >= len(self.query): return True

        return False
        # ### END CODE HERE ###

    def succAndCost(self, state):
        pass
        # ### START CODE HERE ###
        
        result = []
        lwrd = state[1]
        #iterate over state to find the next potential word
        for nxstate in range(state[0]+1,len(self.query)+1):
            wrd = self.query[state[0]:nxstate]
            rfills = self.possibleFills(wrd)
            nxwrd = self.query[state[0]:(nxstate+1)]
            nxfills = self.possibleFills(nxwrd)
            if rfills != set() and nxfills == set(): break
        
        #build the return options
        for rfill in rfills:
            result.append((rfill, (nxstate,rfill),self.bigramCost(lwrd,rfill)))
        return result
        
        # ### END CODE HERE ###


def segmentAndInsert(
    query: str,
    bigramCost: Callable[[str, str], float],
    possibleFills: Callable[[str], Set[str]],
) -> str:
    if len(query) == 0:
        return ""

    # ### START CODE HERE ###
    if len(query) == 0: return ""
    ucs = util.UniformCostSearch(verbose = 0)
    ucs.solve(JointSegmentationInsertionProblem(query,bigramCost,possibleFills))

    output = ""
    for action in ucs.actions:
        output = output + action
        if action != ucs.actions[-1]:
            output = output + " "
    return output
    # ### END CODE HERE ###


############################################################

if __name__ == "__main__":
    shell.main()
