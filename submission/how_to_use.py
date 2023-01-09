import hex_engine as hex
from CAMP_module import agent

if __name__ == "__main__":


    game = hex.hexPosition(7)
    # get action tuple like this:
    ## game: board state as hexPosition
    action = agent(game)
    

    # this only works with our custom changes to hex_engine
    #human_player = 1
    #game.humanVersusMachine(human_player,machine=lambda board: agent(board))