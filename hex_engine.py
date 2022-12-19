class hexPosition(object):
    """
    The class hexPosition stores data on a hex board position. The slots of an object are: size (an integer between 2 and 26), board (an array, 0=noStone, 1=whiteStone, 2=blackStone), and winner (0=noWin, 1=whiteWin, 2=blackWin).
    """

    def __init__(self, size=5):

        if size > 9:
            print("Warning: Large board size, position evaluation may be slow.")

        self.size = max(2, min(size, 26))

        self.board = [[0 for x in range(max(2, min(size, 26)))] for y in range(max(2, min(size, 26)))]
        self.winner = 0

    def reset(self):
        """
        This method resets the hex board. All stones are removed from the board.
        """

        self.board = [[0 for x in range(self.size)] for y in range(self.size)]
        self.winner = 0

    def printBoard(self, invert_colors=True):
        """
        This method prints a visualization of the hex board to the standard output. If the standard output prints black text on a white background, one must set invert_colors=False.
        """

        names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        indent = 0
        headings = " " * 5 + (" " * 3).join(names[:self.size])
        print(headings)
        tops = " " * 5 + (" " * 3).join("_" * self.size)
        print(tops)
        roof = " " * 4 + "/ \\" + "_/ \\" * (self.size - 1)
        print(roof)

        # Attention: Color mapping inverted by default for display in terminal.
        if invert_colors:
            color_mapping = lambda i: " " if i == 0 else ("\u25CB" if i == 2 else "\u25CF")
        else:
            color_mapping = lambda i: " " if i == 0 else ("\u25CF" if i == 2 else "\u25CB")

        for r in range(self.size):
            row_mid = " " * indent
            row_mid += "   | "
            row_mid += " | ".join(map(color_mapping, self.board[r]))
            row_mid += " | {} ".format(r + 1)
            print(row_mid)
            row_bottom = " " * indent
            row_bottom += " " * 3 + " \\_/" * self.size
            if r < self.size - 1:
                row_bottom += " \\"
            print(row_bottom)
            indent += 2
        headings = " " * (indent - 2) + headings
        print(headings)

    def getAdjacent(self, position):
        """
        Helper function to obtain adjacent cells in the board array.
        """

        u = (position[0] - 1, position[1])
        d = (position[0] + 1, position[1])
        r = (position[0], position[1] - 1)
        l = (position[0], position[1] + 1)
        ur = (position[0] - 1, position[1] + 1)
        dl = (position[0] + 1, position[1] - 1)

        return [pair for pair in [u, d, r, l, ur, dl] if
                max(pair[0], pair[1]) <= self.size - 1 and min(pair[0], pair[1]) >= 0]

    def getActionSpace(self, recodeBlackAsWhite=False):
        """
        This method returns a list of array positions which are empty (on which stones may be put).
        """

        actions = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    actions.append((i, j))
        if recodeBlackAsWhite:
            return [self.recodeCoordinates(action) for action in actions]
        else:
            return (actions)

    def playRandom(self, player):
        """
        This method returns a uniformly randomized valid moove for the chosen player (player=1, or player=2).
        """
        from random import choice

        chosen = choice(self.getActionSpace())
        self.board[chosen[0]][chosen[1]] = player

    def randomMatch(self, evaluate_when_full=False):
        """
        This method randomizes an entire playthrough. Mostly useful to test code functionality. If evaluate_when_full=True then the board will be completely filled before the position is evaluated. Otherwise evaluation happens after every moove.
        """

        player = 1

        if evaluate_when_full:
            for i in range(self.size ** 2):
                self.playRandom(player)
                if (player == 1):
                    self.whiteWin()
                    player = 2
                else:
                    self.blackWin()
                    player = 1
            self.whiteWin()
            self.blackWin()

        while self.winner == 0:
            self.playRandom(player)
            if (player == 1):
                self.whiteWin()
                player = 2
            else:
                self.blackWin()
                player = 1

    def prolongPath(self, path):
        """
        A helper function used for board evaluation.
        """

        from copy import deepcopy

        player = self.board[path[-1][0]][path[-1][1]]
        candidates = self.getAdjacent(path[-1])
        candidates = list(filter(lambda cand: cand not in path, candidates))
        candidates = list(filter(lambda cand: self.board[cand[0]][cand[1]] == player, candidates))

        return [deepcopy(path) + [cand] for cand in candidates]

    def whiteWin(self, verbose=False):
        """
        Evaluate whether the board position is a win for 'white'. Uses breadth first search. If verbose=True a winning path will be printed to the standard output (if one exists). This method may be time-consuming, especially for larger board sizes.
        """

        paths = []
        for i in range(self.size):
            if self.board[i][0] == 1:
                paths.append([(i, 0)])

        while True:

            if len(paths) == 0:
                return False

            for path in paths:

                prolongations = self.prolongPath(path)
                paths.remove(path)

                for new in prolongations:
                    if new[-1][1] == self.size - 1:
                        if verbose:
                            print("A winning path for White:\n", new)
                        self.winner = 1
                        return True
                    paths.append(new)

    def blackWin(self, verbose=False):
        """
        Evaluate whether the board position is a win for 'black'. Uses breadth first search. If verbose=True a winning path will be printed to the standard output (if one exists). This method may be time-consuming, especially for larger board sizes.
        """

        paths = []
        for i in range(self.size):
            if self.board[0][i] == 2:
                paths.append([(0, i)])

        while True:

            if len(paths) == 0:
                return False

            for path in paths:

                prolongations = self.prolongPath(path)
                paths.remove(path)

                for new in prolongations:
                    if new[-1][0] == self.size - 1:
                        if verbose:
                            print("A winning path for Black:\n", new)
                        self.winner = 2
                        return True
                    paths.append(new)

    def humanVersusMachine(self, human_player=1, machine=None):
        """
        Play a game against an AI. The variable machine must point to a function that maps a state vector and an action set to an element of the action set. If machine is not specified random actions will be used.
        """

        if machine == None:
            def machine():
                import torch
                from CNN import getActionCNN
                CNN = torch.load('models/model-1671195726.pt').cpu()
                self.board = self.recodeBlackAsWhite(printBoard=False)
                action = getActionCNN(CNN, self, "cpu", self.size, exploit=True)
                self.board = self.recodeBlackAsWhite(printBoard=False)
                action_inv = self.recodeCoordinates(action)
                #print(action_inv)
                return action_inv



        self.reset()

        def translator(string):
            # This function translates human terminal input into the proper array indices.

            number_translated = 27
            letter_translated = 27

            names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

            if len(string) > 0:
                letter = string[0]
            if len(string) > 1:
                number1 = string[1]
            if len(string) > 2:
                number2 = string[2]

            for i in range(26):
                if names[i] == letter:
                    letter_translated = i
                    break

            if len(string) > 2:
                for i in range(10, 27):
                    if number1 + number2 == "{}".format(i):
                        number_translated = i - 1
            else:
                for i in range(1, 10):
                    if number1 == "{}".format(i):
                        number_translated = i - 1

            return (number_translated, letter_translated)

        while self.winner == 0:

            self.printBoard()

            possible_actions = self.getActionSpace()

            human_input = (27, 27)

            while human_input not in possible_actions:
                human_input = translator(input("Enter your moove (e.g. 'A1'): "))

            self.board[human_input[0]][human_input[1]] = 1

            self.whiteWin()

            if self.winner == 1:
                self.printBoard()
                print("The human player (White) has won!")
                self.whiteWin(verbose=True)
            else:
                blacks_moove = machine()
                self.board[blacks_moove[0]][blacks_moove[1]] = 2

                self.blackWin()
                if self.winner == 2:
                    self.printBoard()
                    print("The computer (Black) has won!")
                    self.blackWin(verbose=True)

    def recodeBlackAsWhite(self, printBoard=False, invert_colors=True):
        """
        Returns a board where black is recoded as white and wants to connect horizontally. This corresponds to flipping the board long the south-west to north-east diagonal.
        """
        flipped_board = [[0 for i in range(self.size)] for i in range(self.size)]

        # flipping and color change
        for i in range(self.size):
            for j in range(self.size):
                if self.board[self.size - 1 - j][self.size - 1 - i] == 1:
                    flipped_board[i][j] = 2
                if self.board[self.size - 1 - j][self.size - 1 - i] == 2:
                    flipped_board[i][j] = 1

        names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        indent = 0
        headings = " " * 5 + (" " * 3).join(names[:self.size])
        if printBoard: print(headings)
        tops = " " * 5 + (" " * 3).join("_" * self.size)
        if printBoard: print(tops)
        roof = " " * 4 + "/ \\" + "_/ \\" * (self.size - 1)
        if printBoard: print(roof)

        # Attention: Color mapping inverted by default for display in terminal.
        if invert_colors:
            color_mapping = lambda i: " " if i == 0 else ("\u25CB" if i == 2 else "\u25CF")
        else:
            color_mapping = lambda i: " " if i == 0 else ("\u25CF" if i == 2 else "\u25CB")

        for r in range(self.size):
            row_mid = " " * indent
            row_mid += "   | "
            row_mid += " | ".join(map(color_mapping, flipped_board[r]))
            row_mid += " | {} ".format(r + 1)
            if printBoard: print(row_mid)
            row_bottom = " " * indent
            row_bottom += " " * 3 + " \\_/" * self.size
            if r < self.size - 1:
                row_bottom += " \\"
            if printBoard: print(row_bottom)
            indent += 2
        headings = " " * (indent - 2) + headings
        if printBoard: print(headings)
        return flipped_board

    def recodeCoordinates(self, coordinates):
        """
        Transforms a coordinate tuple (with respect to the board) analogously to the method recodeBlackAsWhite.
        """
        return (self.size - 1 - coordinates[1], self.size - 1 - coordinates[0])

    def coordinate2scalar(self, coordinates):
        """
        Helper function to convert coordinates to scalars.
        """
        return coordinates[0] * self.size + coordinates[1]

    def scalar2coordinates(self, scalar):
        """
        Helper function to transform a scalar "back" to coordinates.
        """
        temp = int(scalar / self.size)
        return (temp, scalar - temp * self.size)

# #Initializing an object
# myboard = hexPosition(size=4)
# #Display the board in standard output
# myboard.printBoard()
# #Random playthrough
# myboard.randomMatch(evaluate_when_full=False)
# myboard.printBoard()
# myboard.blackWin(verbose=True)
# myboard.whiteWin(verbose=True)

# myboard.recodeBlackAsWhite(printBoard=True)

# #check whether Black has won
# myboard.blackWin(verbose=True)
# #check whether White has won
# myboard.whiteWin(verbose=True)
# #print board with inverted colors
# myboard.getInvertedBoard()
# #get board as vector
# myboard.getStateVector(inverted=False)
# #reset the board
# myboard.reset()
# #play against random player
# myboard.humanVersusMachine()
