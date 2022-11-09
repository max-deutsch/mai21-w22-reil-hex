from hex_engine import *

def main():
    #Initializing an object
    myboard = hexPosition(size=7)
    #Display the board in standard output
    myboard.printBoard()
    #Random playthrough
    myboard.randomMatch(evaluate_when_full=False)
    myboard.printBoard()
    #check whether Black has won
    myboard.blackWin(verbose=True)
    #check whether White has won
    myboard.whiteWin(verbose=True)
    #print board with inverted colors
    myboard.getInvertedBoard()
    #get board as vector
    myboard.getStateVector(inverted=False)
    #reset the board
    myboard.reset()
    #play against random player
    myboard.humanVersusMachine()


if __name__ == "__main__":
    main()