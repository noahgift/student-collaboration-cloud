import gc
import util
from util import *
import os, sys
import datetime, time
import argparse
import copy


class SokobanState:
    # player: 2-tuple representing player location (coordinates)
    # boxes: list of 2-tuples indicating box locations
    def __init__(self, player, boxes):
        # self.data stores the state
        self.data = tuple([player] + sorted(boxes))
        # below are cache variables to avoid duplicated computation
        self.all_adj_cache = None
        self.adj = {}
        self.dead = None
        self.solved = None

    def __str__(self):
        return 'player: ' + str(self.player()) + ' boxes: ' + str(self.boxes())

    def __eq__(self, other):
        return type(self) == type(other) and self.data == other.data

    def __lt__(self, other):
        return self.data < other.data

    def __hash__(self):
        return hash(self.data)

    # return player location
    def player(self):
        return self.data[0]

    # return boxes locations
    def boxes(self):
        return self.data[1:]

    def is_goal(self, problem):
        if self.solved is None:
            self.solved = all(problem.map[b[0]][b[1]].target for b in self.boxes())
        return self.solved

    def act(self, problem, act):
        if act in self.adj:
            return self.adj[act]
        else:
            val = problem.valid_move(self, act)
            self.adj[act] = val
            return val

    def deadp(self, problem):
        if self.dead is None:
            for box in self.data[1:]:
                row = box[0]
                col = box[1]
                if problem.dead_map[row][col] == 1:
                    self.dead = 1
                    return self.dead
                else:
                    self.dead = 0
        return self.dead

    def all_adj(self, problem):
        if self.all_adj_cache is None:
            succ = []
            for move in 'udlr':
                valid, box_moved, nextS = self.act(problem, move)
                if valid:
                    succ.append((move, nextS, 1))
            self.all_adj_cache = succ
        return self.all_adj_cache

    def all_box(self, problem):
        def find_next(nextS, passS):
            tempS = nextS
            for move in 'udlr':
                valid, box_moved, nextS = tempS.act(problem, move)
                if valid:
                    if nextS.data not in passS and nextS.deadp(problem) != 1:
                        passS.append(nextS.data)
                        if not box_moved:
                            find_next(nextS, passS)
                        else:
                            succ.append((nextS, nextS, 1))

            return succ

        if self.all_adj_cache is None:
            succ = []

            current_state = SokobanState(player=self.player(), boxes=self.boxes())
            passS = [current_state.data]
            self.all_adj_cache = find_next(current_state, passS)

        return self.all_adj_cache


class MapTile:
    def __init__(self, wall=False, floor=False, target=False):
        self.wall = wall
        self.floor = floor
        self.target = target


def parse_move(move):
    if move == 'u':
        return (-1, 0)
    elif move == 'd':
        return (1, 0)
    elif move == 'l':
        return (0, -1)
    elif move == 'r':
        return (0, 1)
    raise Exception('Invalid move character.')


class DrawObj:
    WALL = '\033[37;47m \033[0m'
    PLAYER = '\033[97;40m@\033[0m'
    BOX_OFF = '\033[30;101mX\033[0m'
    BOX_ON = '\033[30;102mX\033[0m'
    TARGET = '\033[97;40m*\033[0m'
    FLOOR = '\033[30;40m \033[0m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class SokobanProblem(util.SearchProblem):
    # valid sokoban characters
    valid_chars = '#@+$*. '

    def __init__(self, map, dead_detection=False):
        self.map = [[]]
        self.dead_detection = dead_detection
        self.init_player = (0, 0)
        self.init_boxes = []
        self.numboxes = 0
        self.targets = []
        self.parse_map(map)
        self.dead_map = [[]]
        self.dead()
        self.pass_state = []

    # parse the input string into game map
    # Wall              #
    # Player            @
    # Player on target  +
    # Box               $
    # Box on target     *
    # Target            .
    # Floor             (space)
    def parse_map(self, input_str):
        coordinates = lambda: (len(self.map) - 1, len(self.map[-1]) - 1)
        for c in input_str:
            if c == '#':
                self.map[-1].append(MapTile(wall=True))
            elif c == ' ':
                self.map[-1].append(MapTile(floor=True))
            elif c == '@':
                self.map[-1].append(MapTile(floor=True))
                self.init_player = coordinates()
            elif c == '+':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_player = coordinates()
                self.targets.append(coordinates())
            elif c == '$':
                self.map[-1].append(MapTile(floor=True))
                self.init_boxes.append(coordinates())
            elif c == '*':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_boxes.append(coordinates())
                self.targets.append(coordinates())
            elif c == '.':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.targets.append(coordinates())
            elif c == '\n':
                self.map.append([])
        assert len(self.init_boxes) == len(self.targets), 'Number of boxes must match number of targets.'
        self.numboxes = len(self.init_boxes)

    def print_state(self, s):
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                target = self.map[row][col].target
                box = (row, col) in s.boxes()
                player = (row, col) == s.player()
                if box and target:
                    print(DrawObj.BOX_ON, end='')
                elif player and target:
                    print(DrawObj.PLAYER, end='')
                elif target:
                    print(DrawObj.TARGET, end='')
                elif box:
                    print(DrawObj.BOX_OFF, end='')
                elif player:
                    print(DrawObj.PLAYER, end='')
                elif self.map[row][col].wall:
                    print(DrawObj.WALL, end='')
                else:
                    print(DrawObj.FLOOR, end='')
            print()

    # decide if a move is valid
    # return: (whether a move is valid, whether a box is moved, the next state)
    def valid_move(self, s, move, p=None):
        if p is None:
            p = s.player()
        dx, dy = parse_move(move)
        x1 = p[0] + dx
        y1 = p[1] + dy
        x2 = x1 + dx
        y2 = y1 + dy
        if self.map[x1][y1].wall:
            return False, False, None
        elif (x1, y1) in s.boxes():
            if self.map[x2][y2].floor and (x2, y2) not in s.boxes():
                return True, True, SokobanState((x1, y1),
                                                [b if b != (x1, y1) else (x2, y2) for b in s.boxes()])
            else:
                return False, False, None
        else:
            return True, False, SokobanState((x1, y1), s.boxes())

    ## dead_end detection

    def dead(self):
        live_location = []
        for target in self.targets:
            live_location.append(target)
        index = 0
        while index != len(live_location):
            row, col = live_location[index]
            for move in "udlr":
                if move == "u" and row - 2 > 0 and 0 < col < len(self.map[row-2]):
                    if not self.map[row - 2][col].wall and not self.map[row - 1][col].wall and (row - 1, col) not in live_location:
                        live_location.append((row - 1, col))
                elif move == "d" and row + 2 < len(self.map) and 0 < col < len(self.map[row+2]):
                    if not self.map[row + 2][col].wall and not self.map[row + 1][col].wall and (row + 1, col) not in live_location:
                        live_location.append((row + 1, col))
                elif move == "l" and col - 2 > 0 and 0 < row < len(self.map):
                    if not self.map[row][col - 2].wall and not self.map[row][col - 1].wall and (row, col - 1) not in live_location:
                        live_location.append((row, col - 1))
                elif move == "r" and col + 2 < len(self.map[row]) and 0 < row < len(self.map):
                    if not self.map[row][col + 2].wall and not self.map[row][col + 1].wall and (row, col + 1) not in live_location:
                        live_location.append((row, col + 1))
            index += 1

        newmap = copy.deepcopy(self.map)
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):

                if (row, col) in live_location:
                    newmap[row][col] = 0
                else:
                    newmap[row][col] = 1
        self.dead_map = newmap

    # detect dead end
    def dead_end(self, s):
        if not self.dead_detection:
            return False

        return s.deadp(self)

    def start(self):
        return SokobanState(self.init_player, self.init_boxes)

    def goalp(self, s):
        return s.is_goal(self)

    def expand(self, s):
        if self.dead_end(s):
            return []
        return s.all_adj(self)


    ##action compression

class SokobanProblemFaster(SokobanProblem):

    def expand(self, s):
        # return tuples of (box push, next state, cost)
        if self.dead_end(s):
            return []
        result = s.all_box(self)
        return result


    #Simple admissible heuristic

class Heuristic:
    def __init__(self, problem):
        self.problem = problem
        from collections import defaultdict
        self.result = defaultdict(list)
        self.create_matrix()

    def heuristic(self, s):
        total_distance = 0
        for box in s.data[1:]:
            if box not in self.problem.targets:
                temp = 10000
                for tar in self.problem.targets:
                    temp1 = abs(box[0] - tar[0]) + abs(box[1] - tar[1])
                    if temp1 < temp:
                        temp = temp1
                total_distance += temp
        return total_distance

    ##############################################################################
    # Problem 4: Better heuristic.                                               #
    # Implement a better and possibly more complicated heuristic that need not   #
    # always be admissible, but improves the search on more complicated Sokoban  #
    # levels most of the time. Feel free to make any changes anywhere in the     # # code. Our heuristic does some significant work at problem initialization   #
    # and caches it.                                                             #
    # Our solution to this problem affects or adds approximately 40 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################
    def create_matrix(self):
      def recur_matrix(current_pos,past_pos,total_distance,box_loc):
        if current_pos not in past_pos and 0 <= current_pos[0] < len(self.problem.map) and 0 <=current_pos[1] < len(self.problem.map[current_pos[0]]):
          past_pos.append(current_pos)
          if not self.problem.map[current_pos[0]][current_pos[1]].wall:
            if current_pos not in self.problem.targets:
              recur_matrix((current_pos[0]-1,current_pos[1]),past_pos,total_distance+1,box_loc)
              recur_matrix((current_pos[0]+1,current_pos[1]),past_pos,total_distance+1,box_loc)
              recur_matrix((current_pos[0],current_pos[1]-1),past_pos,total_distance+1,box_loc)
              recur_matrix((current_pos[0],current_pos[1]+1),past_pos,total_distance+1,box_loc)
            else:
             self.result[box_loc].append(((current_pos[0],current_pos[1]),total_distance))

      for row in range(len(self.problem.map)):
        for col in range(len(self.problem.map[row])):
          if not self.problem.map[row][col].wall:
            past_pos = []
            recur_matrix((row, col), past_pos, 0, (row, col))

    def heuristic2(self, s):
        box_notin = []
        for box in s.data[1:]:
            if box not in self.problem.targets:
                box_notin.append(self.result[box])
        total_distance = self.find_least(box_notin)
        return total_distance

    def find_least(self, notin):
      temp = 0
      for box in notin:
        temp1 = []
        for target in box:
          temp1.append(target[1])
        temp += min(temp1)
      return temp


# solve sokoban map using specified algorithm

def find_path(problem, prevS, newS, prevmove, passS):
    pos1 = prevS.data[0]
    pos2 = newS.data[0]
    verse = {"u":"d", "d":"u", "l":"r", "r":"l"}
    if pos1[0] <= pos2[0]:
        if pos1[1] <= pos2[1]:  # right-down corner
            turn = "drlu"
        else:  # left-down corner
            turn = "dlur"
    else:
        if pos1[1] <= pos2[1]:  # right-up corner
            turn = "urld"
        else:  # left-up corner
            turn = "uldr"
    for move in turn:
        valid, box_moved, state = problem.valid_move(prevS, move)
        if valid and state not in passS and state.deadp(problem) != 1:
            if state == newS:
                passS.append(state)
                return move
            else:
                if box_moved != 1:
                    if len(prevmove) == 0 or (len(prevmove) > 0 and prevmove != verse[move]):
                        passS.append(state)
                        new_move = find_path(problem, state, newS, move, passS)
                        if new_move:
                            return move+new_move


def solve_sokoban(map, algorithm='ucs', dead_detection=False):
    # problem algorithm
    if 'f' in algorithm:
        problem = SokobanProblemFaster(map, dead_detection)
    else:
        problem = SokobanProblem(map, dead_detection)

    # search algorithm
    h = Heuristic(problem).heuristic2 if ('2' in algorithm) else Heuristic(problem).heuristic
    if 'a' in algorithm:
        search = util.AStarSearch(heuristic=h)
    else:
        search = util.UniformCostSearch()

    # solve problem
    search.solve(problem)
    # if search.actions is not None:
    #     print('length {} soln is {}'.format(len(search.actions), search.actions))
    if 'f' in algorithm:
        actions = []
        prevS = problem.start()
        ct = 0
        for action in search.actions:
            passS = [prevS]
            path = ""
            final_path = path + find_path(problem, prevS, action, path, passS)
            prevS = action
            ct += 1
            actions.append(final_path)

        return search.totalCost, actions, search.numStatesExplored
    else:
        return search.totalCost, search.actions, search.numStatesExplored


# animate the sequence of actions in sokoban map
def animate_sokoban_solution(map, seq, dt=0.2):
    problem = SokobanProblem(map)
    state = problem.start()
    clear = 'cls' if os.name == 'nt' else 'clear'
    for i in range(len(seq)):
        os.system(clear)
        print(seq[:i] + DrawObj.UNDERLINE + seq[i] + DrawObj.END + seq[i + 1:])
        problem.print_state(state)
        time.sleep(dt)
        valid, _, state = problem.valid_move(state, seq[i])
        if not valid:
            raise Exception('Cannot move ' + seq[i] + ' in state ' + str(state))
    os.system(clear)
    print(seq)
    problem.print_state(state)


# read level map from file, returns map represented as string
def read_map_from_file(file, level):
    map = ''
    start = False
    found = False
    with open(file, 'r') as f:
        for line in f:
            if line[0] == "'": continue
            if line.strip().lower()[:5] == 'level':
                if start: break
                if line.strip().lower() == 'level ' + level:
                    found = True
                    start = True
                    continue
            if start:
                if line[0] in SokobanProblem.valid_chars:
                    map += line
                else:
                    break
    if not found:
        raise Exception('Level ' + level + ' not found')
    return map.strip('\n')


# extract all levels from file
def extract_levels(file):
    levels = []
    with open(file, 'r') as f:
        for line in f:
            if line.strip().lower()[:5] == 'level':
                levels += [line.strip().lower()[6:]]
    return levels


def solve_map(file, level, algorithm, dead, simulate):
    map = read_map_from_file(file, level)
    print(map)
    tic = datetime.datetime.now()
    cost, sol, nstates = solve_sokoban(map, algorithm, dead)
    toc = datetime.datetime.now()
    print('Time consumed: {:.3f} seconds using {} and exploring {} states'.format(
        (toc - tic).seconds + (toc - tic).microseconds / 1e6, algorithm, nstates))
    seq = ''.join(sol)
    print(len(seq), 'moves')
    print(' '.join(seq[i:i + 5] for i in range(0, len(seq), 5)))
    if simulate:
        animate_sokoban_solution(map, seq)


def main():
    parser = argparse.ArgumentParser(description="Solve Sokoban map")
    parser.add_argument("level", help="Level name or 'all'")
    parser.add_argument("algorithm", help="ucs | [f][a[2]] | all")
    parser.add_argument("-d", "--dead", help="Turn on dead state detection (default off)", action="store_true")
    parser.add_argument("-s", "--simulate", help="Simulate the solution (default off)", action="store_true")
    parser.add_argument("-f", "--file", help="File name storing the levels (levels.txt default)", default='levels.txt')
    parser.add_argument("-t", "--timeout", help="Seconds to allow (default 300)", type=int, default=300)

    args = parser.parse_args()
    level = args.level
    algorithm = args.algorithm
    dead = args.dead
    simulate = args.simulate
    file = args.file
    maxSeconds = args.timeout

    if (algorithm == 'all' and level == 'all'):
        raise Exception('Cannot do all levels with all algorithms')

    def solve_now():
        solve_map(file, level, algorithm, dead, simulate)

    def solve_with_timeout(maxSeconds):
        try:
            util.TimeoutFunction(solve_now, maxSeconds)()
        except KeyboardInterrupt:
            raise
        except MemoryError as e:
            signal.alarm(0)
            gc.collect()
            print('Memory limit exceeded.')
        except util.TimeoutFunctionException as e:
            signal.alarm(0)
            print('Time limit (%s seconds) exceeded.' % maxSeconds)

    if level == 'all':
        levels = extract_levels(file)
        for level in levels:
            print('Starting level {}'.format(level), file=sys.stderr)
            sys.stdout.flush()
            solve_with_timeout(maxSeconds)
    elif algorithm == 'all':
        for algorithm in ['ucs', 'a', 'a2', 'f', 'fa', 'fa2']:
            print('Starting algorithm {}'.format(algorithm), file=sys.stderr)
            sys.stdout.flush()
            solve_with_timeout(maxSeconds)
    else:
        solve_with_timeout(maxSeconds)


if __name__ == '__main__':
    main()
