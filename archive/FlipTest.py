import hex_engine

def test1():
    print("---- Test 1 -----")
    game_state = hex_engine.hexPosition(2)

    game_state.board[0][0] = 1

    game_state.printBoard()
    print("recodeBlackAsWhite()")
    game_state.board = game_state.recodeBlackAsWhite()
    game_state.printBoard()
    print("-----------------")


def test2():
    print("---- Test 2 -----")
    game_state = hex_engine.hexPosition(2)

    game_state.board[0][0] = 1
    game_state.board[1][1] = 2

    game_state.printBoard()
    print("recodeBlackAsWhite()")
    game_state.board = game_state.recodeBlackAsWhite()
    game_state.printBoard()
    print("----------------")


if __name__ == "__main__":
    test1()
    test2()
