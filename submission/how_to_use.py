import hex_engine as hex
import CAMP_module as CAMP

if __name__ == "__main__":

    player1 = CAMP.hexCAMP('CAMP_model.pt')
    game = hex.hexPosition(7)
    # get action tuple like this:
    ## game: board state as hexPosition
    ## player: player= [ 1 , 2 ], i.e. white or black
    action = player1.play(game=game, player=1)

    CAMP.evalVSrandom(player1,1000)

    # this only works with our custom changes to hex_engine
    #human_player = 1
    #game.humanVersusMachine(human_player,machine=lambda board: player1.play(board, 2 if human_player == 1 else 1))