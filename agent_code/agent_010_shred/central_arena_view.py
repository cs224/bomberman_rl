
import collections, heapq, time, logging
import numpy as np, pandas as pd, xarray as xr
from settings import s
import numba

log = logging.getLogger('central_arena_view')

class CentralArenaView():

    NX = 29
    NY = 29

    def __init__(self, *args, **kwargs):
        self.rotation            =  0
        self.mirror              =  1
        self.code_wall           = -1
        self.code_free           =  0
        self.code_crate          =  1
        self.code_coin           =  2
        self.code_me             =  3
        self.code_other_agent    =  4
        self.code_bomb           =  5
        self.code_bomb_base      = 10
        self.code_bomb_explosion = 50

        self.bomb_tmax           = s.bomb_timer + 1

        self.x_center = 14
        self.y_center = 14

        self.delta_x = 15
        self.delta_y = 15

        self.central_arena_view = np.full((CentralArenaView.NY, CentralArenaView.NX), -1, np.byte)

        self.x_new = -1
        self.y_new = -1

        if len(args) == 0:
            return

        agent_game_state = args[0]
        self.agent_game_state    = agent_game_state

        original_arena, agent_self = agent_game_state['arena'], agent_game_state['self']

        a_x = agent_self[0] - 1
        a_y = agent_self[1] - 1
        x_new = self.x_center - a_x
        y_new = self.y_center - a_y
        self.central_arena_view[y_new:(y_new + self.delta_y), x_new:(x_new + self.delta_x)] = original_arena.T[1:-1, 1:-1]
        self.x_new = x_new
        self.y_new = y_new

        # do not set a code for yourself, as you're anyway always in the middle!
        # putting yourself there will only hide a potential bomb below you that you put yourself
        # self.set_code_at_coordinate(agent_self[0], agent_self[1], self.code_me)

        self.add_other_agents()
        self.add_coins()
        self.add_bombs()
        self.add_explosions()

    def copy_from(self, other):
        self.agent_game_state        = None
        self.rotation                = other.rotation
        self.mirror                  = other.mirror
        self.central_arena_view[:,:] = other.central_arena_view[:,:]
        self.x_new                   = other.x_new
        self.y_new                   = other.y_new

    def central_arena_view_coordinates_from_original_coordinates(self, x, y):
        return (self.x_new + x - 1, self.y_new + y - 1)

    def get_code_at_coordinate(self, x, y):
        nx,ny = self.central_arena_view_coordinates_from_original_coordinates(x, y)
        return self.central_arena_view[ny, nx]

    def set_code_at_coordinate(self, x, y, code):
        nx,ny = self.central_arena_view_coordinates_from_original_coordinates(x, y)
        self.central_arena_view[ny, nx] = code

    def get(self, x, y):
        return self.central_arena_view[y, x]

    def set(self, x, y, code):
        self.central_arena_view[y, x] = code

    def add_other_agents(self):
        other_agents = self.agent_game_state['others']
        for a in other_agents:
            self.add_other_agent(a)

    def add_other_agent(self, a):
        self.set_code_at_coordinate(a[0], a[1], self.code_other_agent)

    def add_coins(self):
        coins = self.agent_game_state['coins']
        for coin in coins:
            x,y = coin
            self.set_code_at_coordinate(x, y, self.code_coin)

    def add_bombs(self):
        bombs = self.agent_game_state['bombs']
        # XXX in principle I would need to see which bombs are in reach of each other and set their timers to the minimum timer
        for bomb in bombs:
            self.add_bomb(bomb)

    def calc_bomb_field_code(self, base_code, tmax, t):
        code = self.code_bomb_base
        code += base_code * tmax + t
        return code

    def add_bomb(self, bomb):
        tmax = self.bomb_tmax
        x_,y_, t =  bomb
        self.set_code_at_coordinate(x_, y_, self.calc_bomb_field_code(self.code_bomb, tmax, t))

        blast_coordinates = self.get_bomb_blast_coords(x_, y_)

        for bc in blast_coordinates:
            x,y = bc
            base_code = self.get_code_at_coordinate(x,y)
            if base_code >= self.code_bomb_base: # do not add several times the values
                continue
            # the logic below is to mark fields that will be deadly already when you know that, e.g. as soon as the bomb is on the game arena
            # in addition you want to distinguish coins that will explode in 2 rounds from coins that will explode in 1 round and so on,
            # so that the agent has the chance to act differently based on this information
            self.set_code_at_coordinate(x, y, self.calc_bomb_field_code(base_code, tmax, t))

    # self.code_free           =  0
    # self.code_crate          =  1
    # self.code_coin           =  2
    # self.code_me             =  3
    # self.code_other_agent    =  4
    # self.code_bomb           =  5

    def is_code_p(self, np_array, base_code):
        tmax = self.bomb_tmax
        is_exact  = (np_array == base_code)
        is_bombed = (np_array >= self.code_bomb_base + base_code * tmax) & (np_array < self.code_bomb_base + (base_code+1) * tmax)

        return is_exact | is_bombed

    def is_code(self, np_array, base_code):
        tmax = self.bomb_tmax
        is_exact  = (np_array == base_code)
        is_bombed = (np_array >= self.code_bomb_base + base_code * tmax) & (np_array < self.code_bomb_base + (base_code+1) * tmax)

        return np.where(is_exact | is_bombed)

    def is_code_single_value(self, value, base_code):
        return PACAV.is_code(value, base_code)[0].size > 0

    def get_bomb_blast_coords(self, x, y):
        arena = self.agent_game_state['arena']
        power = s.bomb_power
        blast_coords = [(x,y)]

        for i in range(1, power+1):
            if arena[x+i,y] == -1: break
            blast_coords.append((x+i,y))
        for i in range(1, power+1):
            if arena[x-i,y] == -1: break
            blast_coords.append((x-i,y))
        for i in range(1, power+1):
            if arena[x,y+i] == -1: break
            blast_coords.append((x,y+i))
        for i in range(1, power+1):
            if arena[x,y-i] == -1: break
            blast_coords.append((x,y-i))

        return blast_coords

    def add_explosions(self):
        tmax = s.explosion_timer
        explosions = self.agent_game_state['explosions']
        for x in range(1,16):
            for y in range(1,16):
                explosion_value = explosions[x,y]
                if explosion_value > 0:
                    base_code = self.get_code_at_coordinate(x, y)
                    # XXX most likely the below is superfluous, because explosion is explosion and the base_code does not matter
                    # self.set_code_at_coordinate(x,y, self.code_bomb_explosion + explosion_value - 1 + base_code * tmax)
                    self.set_code_at_coordinate(x,y, self.code_bomb_explosion + explosion_value - 1)

    def calulate_rotate_right_x_new_y_new(self, x_new, y_new):
        old_x_new = x_new
        old_y_new = y_new

        lower_left_x = old_x_new
        lower_left_y = old_y_new + self.delta_y - 1

        x_new = 28 - lower_left_y
        y_new = lower_left_x
        return x_new, y_new


    def rotate_right_(self):
        r = np.full((29, 29), -1, np.byte)
        for i in range(29):
            r[i,:] = self.central_arena_view[::-1,i]

        self.central_arena_view = r
        self.rotation += 1
        self.agent_game_state   = None

        x_new, y_new = self.calulate_rotate_right_x_new_y_new(self.x_new, self.y_new)
        self.y_new = y_new
        self.x_new = x_new

    def rotate_right(self):
        # print('rotate_right: before: {}'.format(self.rotation))
        r = self.copy()
        # r = self.__class__() # CentralArenaView()
        # r.copy_from(self)
        # print('rotate_right: after copy: {}'.format(r.rotation))
        r.rotate_right_()
        # print('rotate_right: after rotation: {}'.format(r.rotation))
        return r

    def mirror_vertical_axis_(self):
        r = self.central_arena_view[:,::-1].copy()

        self.central_arena_view = r
        self.mirror *= -1
        self.agent_game_state   = None

        # mirror right upper corner
        old_x_new = self.x_new
        right_upper_x = old_x_new + self.delta_x - 1
        self.x_new = 28 - right_upper_x

    def mirror_vertical_axis(self):
        r = self.copy()
        # r = self.__class__() # CentralArenaView()
        # r.copy_from(self)
        r.mirror_vertical_axis_()
        return r

    def copy(self):
        # print('copy: {}'.format(self.__class__.__name__))
        r = self.__class__() # CentralArenaView()
        r.copy_from(self)
        # print('copy: {}'.format(type(r).__name__))
        return r


# https://www.redblobgames.com/pathfinding/a-star/introduction.html
# https://www.redblobgames.com/pathfinding/a-star/implementation.html

class Queue:
    def __init__(self):
        self.elements = collections.deque()

    def empty(self):
        return len(self.elements) == 0

    def put(self, x):
        self.elements.append(x)

    def get(self):
        return self.elements.popleft()


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


class SimpleGraph:
    def __init__(self):
        self.edges = {}

    def neighbors(self, id):
        return self.edges[id]


class CentralArenaViewSquareGrid:
    def __init__(self, central_arena_view):
        self.cav = central_arena_view
        self.width  = self.cav.central_arena_view.shape[1]
        self.height = self.cav.central_arena_view.shape[0]
        self.solid_codes = [self.cav.code_wall]

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id):
        (x, y) = id
        code = self.cav.get(x,y)
        return code not in self.solid_codes

    def neighbors(self, id):
        (x, y) = id
        results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
        # if (x + y) % 2 == 0: results.reverse()  # aesthetics
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results

class PathWeights:
    crate       = 6
    other_agent = 4
    explosion   = 2
    bomb        = 3

class CentralArenaViewWithWeights(CentralArenaViewSquareGrid):
    def __init__(self, central_arena_view):
        super().__init__(central_arena_view)
        self.weights = {
            self.cav.code_crate          : PathWeights.crate     , # max: 8; min: 10/3~4
            self.cav.code_other_agent    : PathWeights.other_agent, # may be an obstacle or run away
        }

    def cost(self, from_node, to_node):
        (x, y) = to_node
        field_value = self.cav.get(x,y)
        if field_value > 50:
            return PathWeights.explosion
        if field_value > 10:
            if PACAV.is_code_single_value(field_value, PACAV.code_free):
                return self.weights.get(PACAV.code_free, 1)
            if PACAV.is_code_single_value(field_value, PACAV.code_coin):
                return self.weights.get(PACAV.code_free, 1)
            return PathWeights.bomb # will definitely explode without my doing; a crate needs my doing to go away so its value should be higher
        return self.weights.get(field_value, 1)

class FloodSearch():
    def __init__(self, central_arena_view):
        self.cav = central_arena_view
        self.graph = CentralArenaViewWithWeights(self.cav)
        self.flood()

    def is_crate(self, coords):
        x, y = coords
        field_value = self.cav.central_arena_view[y,x]
        r = field_value == self.cav.code_crate
        # print(field_value, self.cav.code_crate, r)
        return r

    def flood(self):
        start = (14,14)
        frontier = Queue()
        frontier.put(start)
        came_from = {}
        came_from[start] = None
        cost_so_far = {}
        cost_so_far[start] = (0,0)
        direct_path = {}
        direct_path[start] = (True,True)

        nearest_crate = None

        while not frontier.empty():
            current = frontier.get()
            for next in self.graph.neighbors(current):
                node_cost = self.graph.cost(current, next)
                prev_cost = cost_so_far[current][0]
                new_cost  = prev_cost + node_cost

                if next not in cost_so_far or new_cost < cost_so_far[next][0]:
                    if nearest_crate is None and self.is_crate(next):
                        nearest_crate = next

                    cost_so_far[next] = (new_cost, prev_cost)
                    frontier.put(next)
                    came_from[next] = current
                    if current in direct_path:
                        node_is_direct_path_node       = node_cost <= 1
                        path_until_here_is_direct_path = direct_path[current][0]
                        if path_until_here_is_direct_path:
                            direct_path[next] = (node_is_direct_path_node, path_until_here_is_direct_path)

        self.came_from    = came_from
        self.cost_so_far  = cost_so_far
        self.direct_path  = direct_path
        self.nearest_crate = nearest_crate

    def get_path_to(self, x, y):
        current_node = (x,y)
        direct_path = current_node in self.direct_path
        cost        = self.cost_so_far[current_node][1]
        path = [current_node]
        next = self.came_from[current_node]
        while next is not None:
            path = [next] + path
            next = self.came_from[next]

        if len(path) <= 1: # I am already there
            dx = 0
            dy = 0
        else:
            path_element_2 = path[1]
            path_element_1 = path[0]
            dx = path_element_2[0] - path_element_1[0]
            dy = path_element_2[1] - path_element_1[1]
        direction = (dx,dy)

        r = collections.namedtuple("PathSearchResult", ['direction', 'is_direct_path', 'cost', 'path'])(direction, direct_path, cost, path)

        return r


# 2019-02-17 13:00:43,879 [simple_agent_0_wrapper] ERROR: Error in callback function: list index out of range
# Traceback (most recent call last):
#   File "/home/local/cs/workspaces/bomberman_rl/agents.py", line 106, in run
#     self.code.act(self.fake_self)
#   File "/home/local/cs/workspaces/bomberman_rl/agent_code/simple_agent/callbacks.py", line 207, in act
#     self.dch.callback01_register_game_state_and_action(self.game_state, self.next_action)
#   File "/home/local/cs/workspaces/bomberman_rl/bm_data_collection_helper.py", line 65, in callback01_register_game_state_and_action
#     self.av = cav.PandasAugmentedCentralArenaView(game_state)
#   File "/home/local/cs/workspaces/bomberman_rl/central_arena_view.py", line 465, in __init__
#     super().__init__(*args, **kwargs)
#   File "/home/local/cs/workspaces/bomberman_rl/central_arena_view.py", line 369, in __init__
#     self.augment_mid_of_map_data()
#   File "/home/local/cs/workspaces/bomberman_rl/central_arena_view.py", line 423, in augment_mid_of_map_data
#     r = self.fs.get_path_to(nx, ny)
#   File "/home/local/cs/workspaces/bomberman_rl/central_arena_view.py", line 326, in get_path_to
#     path_element_2 = path[1]
# IndexError: list index out of range

class AugmentedCentralArenaView(CentralArenaView):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dmap = {
            ( 0,  0) : -1,
            ( 0, -1) :  0,
            (-1,  0) :  1,
            ( 0,  1) :  2,
            ( 1,  0) :  3,
        }

        r = None
        rd = np.full(3, -1, np.byte)
        self.nearest_other_agent_info = r
        self.nearest_other_agent_info_data = rd

        rd = np.full(3, -1, np.byte)
        self.nearest_coin_info = r
        self.nearest_coin_info_data = rd

        rd = np.full(3, -1, np.byte)
        self.nearest_crate_info = r
        self.nearest_crate_info_data = rd

        rd = np.full(3, -1, np.byte)
        self.mid_of_map_info = r
        self.mid_of_map_info_data = rd

        if len(args) == 0:
            return

        self.fs = FloodSearch(self)
        self.augment_nearest_other_agent_data()
        self.augment_nearest_coin_data()
        self.augment_nearest_crate_data()
        self.augment_mid_of_map_data()


    def extract_object_of_interest_data(self, object_of_interest_list):
        r = None
        rd = np.full(3, -1, np.byte)
        for o in object_of_interest_list:
            x = o[0]
            y = o[1]
            nx,ny = self.central_arena_view_coordinates_from_original_coordinates(x, y)
            p = self.fs.get_path_to(nx,ny)

            if r is None or p.cost < r.cost:
                r = p

        if r is not None:
            rd[0] = self.direction_to_value(r.direction)
            rd[1] = r.is_direct_path
            rd[2] = r.cost

        return r, rd

    def direction_to_value(self, d):
        return self.dmap[d] # should throw an exception in case that d is not in dmap

    def augment_nearest_other_agent_data(self):
        agents = self.agent_game_state['others']
        r, rd = self.extract_object_of_interest_data(agents)
        self.nearest_other_agent_info = r
        self.nearest_other_agent_info_data = rd

    def augment_nearest_coin_data(self):
        coins = self.agent_game_state['coins']
        r, rd = self.extract_object_of_interest_data(coins)
        self.nearest_coin_info = r
        self.nearest_coin_info_data = rd

    def augment_nearest_crate_data(self):
        r = None
        rd = np.full(3, -1, np.byte)
        nearest_crate = self.fs.nearest_crate
        if nearest_crate is not None:
            nx, ny = nearest_crate
            r = self.fs.get_path_to(nx, ny)
            rd[0] = self.direction_to_value(r.direction)
            rd[1] = r.is_direct_path
            rd[2] = r.cost
        self.nearest_crate_info = r
        self.nearest_crate_info_data = rd

    def augment_mid_of_map_data(self):
        rd = np.full(3, -1, np.byte)
        mid_of_map = self.x_new + self.delta_x // 2, self.y_new + self.delta_y // 2 - 1
        nx, ny = mid_of_map
        r = self.fs.get_path_to(nx, ny)
        rd[0] = self.direction_to_value(r.direction)
        rd[1] = r.is_direct_path
        rd[2] = r.cost
        self.mid_of_map_info = r
        self.mid_of_map_info_data = rd

    # the following methods need to be adapted to make sure that they also work correctly for augmented data
    def copy_from(self, other):
        super().copy_from(other)
        # self.nearest_other_agent_info         = other.nearest_other_agent_info
        self.nearest_other_agent_info_data[:] = other.nearest_other_agent_info_data[:]

        # self.nearest_coin_info                = other.nearest_coin_info
        self.nearest_coin_info_data[:]        = other.nearest_coin_info_data[:]

        # self.nearest_crate_info               = other.nearest_crate_info
        self.nearest_crate_info_data[:]       = other.nearest_crate_info_data[:]

        # self.mid_of_map_info                  = other.mid_of_map_info
        self.mid_of_map_info_data             = other.mid_of_map_info_data[:]

    # self.dmap = {
    #     ( 0,  0) : -1,
    #     ( 0, -1) :  0,
    #     (-1,  0) :  1,
    #     ( 0,  1) :  2,
    #     ( 1,  0) :  3,
    # }
    def direction_rotate_right(self, d):
        if d < 0 or d > 3:
            return d

        # 0 -> 3
        # 1 -> 0
        # 2 -> 1
        # 3 -> 2
        return (4 + d - 1) % 4

    def direction_mirror_vertical_axis(self, d):
        if d < 0 or d > 3:
            return d

        # 0 -> 0
        # 1 -> 3
        # 2 -> 2
        # 3 -> 1
        if d == 1:
            return 3

        if d == 3:
            return 1

        return d

    def rotate_right_(self):
        super().rotate_right_()
        self.nearest_other_agent_info_data[0] = self.direction_rotate_right(self.nearest_other_agent_info_data[0])
        self.nearest_coin_info_data[0]        = self.direction_rotate_right(self.nearest_coin_info_data[0])
        self.nearest_crate_info_data[0]       = self.direction_rotate_right(self.nearest_crate_info_data[0])
        self.mid_of_map_info_data[0]          = self.direction_rotate_right(self.mid_of_map_info_data[0])

    def mirror_vertical_axis_(self):
        super().mirror_vertical_axis_()
        self.nearest_other_agent_info_data[0] = self.direction_mirror_vertical_axis(self.nearest_other_agent_info_data[0])
        self.nearest_coin_info_data[0]        = self.direction_mirror_vertical_axis(self.nearest_coin_info_data[0])
        self.nearest_crate_info_data[0]       = self.direction_mirror_vertical_axis(self.nearest_crate_info_data[0])
        self.mid_of_map_info_data[0]          = self.direction_mirror_vertical_axis(self.mid_of_map_info_data[0])

# Data Frame Colum IDs
class DFIDs:
    DataCollectionID      = 'DataCollectionID'
    agent_id              = 'aid'
    game_count            = 'gc'
    ts                    = 'ts'
    game_step             = 'gs'
    ROTATION              = 'rot'

    QQ_Pred               = 'QQ_Pred'
    QQ_Pred_Max           = 'QQ_Pred_Max'
    QQSC_Pred             = 'QQSC_Pred'
    TTL_Pred              = 'TTL_Pred'
    S_Pred                = 'S_Pred'
    W_Pred                = 'W_Pred'
    Pred_dt               = 'Pred_dt'

    Q_CRATE_DESTROYED     = 'QCrD'
    Q_COIN_FOUND          = 'QCoF'
    Q_COIN_COLLECTED      = 'QCoC'
    Q_OPPONENT_ELIMINATED = 'QOE'
    Q_KILLED_OPPONENT     = 'QOK'
    Q_GOT_KILLED          = 'QKO'
    Q_KILLED_SELF         = 'QKS'
    Q_PENALTY             = 'QP'
    Q_SCORE               = 'QSC'
    Q                     = 'Q'
    QQ                    = 'QQ'
    QQ_SCORE              = 'QQSC'
    TTL                   = 'TTL'
    S                     = 'S'
    W                     = 'W'
    DIRECTION             = 'D'
    ACTION                = 'A'
    A_WAIT                = 'A_WAIT'
    A_UP                  = 'A_UP'
    A_LEFT                = 'A_LEFT'
    A_DOWN                = 'A_DOWN'
    A_RIGHT               = 'A_RIGHT'
    A_BOMB                = 'A_BOMB'
    A_ONE_HOT             = ['A_WAIT', 'A_UP', 'A_LEFT', 'A_DOWN', 'A_RIGHT', 'A_BOMB']

    CENTRAL_ARENA_VIEW_OFFSET_X = 'ox'
    CENTRAL_ARENA_VIEW_OFFSET_Y = 'oy'
    CENTRAL_ARENA_VIEW_FORMAT = 'cav_x{:02d}_y{:02d}'

# new variables:
#  add to DFIDs,
#  create xxx_columns and xxx_data in PandasAugmentedCentralArenaView
#  add them to self.df_columns
#  add to copy_from
#  if rotation/mirror dependent make sure to add relevant code to rotate_ and mirror_
#  write the code for to_df
#  write code for from_df


class PandasAugmentedCentralArenaView(AugmentedCentralArenaView):

    auxiliary_variable1_columns = [DFIDs.DataCollectionID, DFIDs.ts]
    auxiliary_variable2_columns = [DFIDs.agent_id]
    auxiliary_variable3_columns = [DFIDs.game_count, DFIDs.game_step]
    rotation_variable_columns = [DFIDs.ROTATION]
    sort_columns = auxiliary_variable1_columns + auxiliary_variable2_columns + auxiliary_variable3_columns + rotation_variable_columns
    prediction_variable_columns = [DFIDs.Pred_dt]  # DFIDs.QQ_Pred, DFIDs.QQ_Pred_Max, DFIDs.QQSC_Pred, DFIDs.TTL_Pred, DFIDs.S_Pred, DFIDs.W_Pred,
    q_variable_columns = [DFIDs.Q_CRATE_DESTROYED, DFIDs.Q_COIN_FOUND, DFIDs.Q_COIN_COLLECTED, DFIDs.Q_OPPONENT_ELIMINATED, DFIDs.Q_KILLED_OPPONENT, DFIDs.Q_GOT_KILLED, DFIDs.Q_KILLED_SELF, DFIDs.Q_PENALTY]
    q_score_columns = [DFIDs.Q_SCORE]
    target_variable_columns = [DFIDs.S, DFIDs.W]
    direction_variable_columns = [DFIDs.DIRECTION, DFIDs.ACTION]
    nearest_other_agent_info_columns = ['noa_direction', 'noa_is_direct_path', 'noa_cost']
    nearest_coin_info_columns = ['nco_direction', 'nco_is_direct_path', 'nco_cost']
    nearest_crate_info_columns = ['ncr_direction', 'ncr_is_direct_path', 'ncr_cost']
    mid_of_map_info_columns = ['mom_direction', 'mom_is_direct_path', 'mom_cost']
    central_arena_view_offset_columns = [DFIDs.CENTRAL_ARENA_VIEW_OFFSET_X, DFIDs.CENTRAL_ARENA_VIEW_OFFSET_Y]
    central_arena_view_columns = [DFIDs.CENTRAL_ARENA_VIEW_FORMAT.format(x, y) for y in range(CentralArenaView.NY) for x in range(CentralArenaView.NX)]

    df_columns = auxiliary_variable1_columns + auxiliary_variable2_columns + auxiliary_variable3_columns + rotation_variable_columns + q_variable_columns + prediction_variable_columns + q_score_columns + \
        target_variable_columns + direction_variable_columns + nearest_other_agent_info_columns + nearest_coin_info_columns + nearest_crate_info_columns + mid_of_map_info_columns + \
        central_arena_view_offset_columns + central_arena_view_columns

    df = pd.DataFrame(columns=df_columns)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 'DataCollectionID' identifies a run of the data collection process
        # 'ts' timestamp
        self.auxiliary_variable1_columns       = PandasAugmentedCentralArenaView.auxiliary_variable1_columns
        self.auxiliary_variable1_data          = np.array([0,0], dtype=np.int32)
        self.auxiliary_variable2_columns       = PandasAugmentedCentralArenaView.auxiliary_variable2_columns
        self.auxiliary_variable2_data          = np.array([0], dtype=np.byte)
        self.auxiliary_variable3_columns       = PandasAugmentedCentralArenaView.auxiliary_variable3_columns
        self.auxiliary_variable3_data          = np.array([0,0], dtype=np.int16)

        self.rotation_variable_columns         = PandasAugmentedCentralArenaView.rotation_variable_columns
        self.rotation_variable_data            = np.array([''], dtype='<U4')

        self.sort_columns                      = PandasAugmentedCentralArenaView.sort_columns

        # XXX originally I thought I'd add the predicted values to the recorded fields during game execution, but this is just bloating the data file
        #     you can always generate these values via running your model on the input data.
        self.prediction_variable_columns       = PandasAugmentedCentralArenaView.prediction_variable_columns
        self.prediction_variable_data          = np.array([0.0] * len(self.prediction_variable_columns), dtype=np.float32)

        self.q_variable_columns                = PandasAugmentedCentralArenaView.q_variable_columns
        self.q_variable_data                   = np.array([0] * 8, dtype=np.byte)

        self.q_score_columns                   = PandasAugmentedCentralArenaView.q_score_columns
        self.q_score_data                      = np.array([0] * 1, dtype=np.byte)

        #   Q: single step reward
        #  QQ: accumulated reward
        # TTL: Time to Live: how many steps until you die
        #   S: Survive: yes or no
        #   W:     Win: yes or no
        self.target_variable_columns          = PandasAugmentedCentralArenaView.target_variable_columns
        self.target_variable_data             = np.array([0,0], dtype=np.byte)

        self.direction_variable_columns       = PandasAugmentedCentralArenaView.direction_variable_columns
        self.direction_variable_data          = np.array([0]*2, dtype=np.byte)

        self.nearest_other_agent_info_columns = PandasAugmentedCentralArenaView.nearest_other_agent_info_columns
        self.nearest_coin_info_columns        = PandasAugmentedCentralArenaView.nearest_coin_info_columns
        self.nearest_crate_info_columns       = PandasAugmentedCentralArenaView.nearest_crate_info_columns
        self.mid_of_map_info_columns          = PandasAugmentedCentralArenaView.mid_of_map_info_columns

        self.central_arena_view_offset_columns = PandasAugmentedCentralArenaView.central_arena_view_offset_columns
        self.central_arena_view_offset_data    = np.array([0]*2, dtype=np.byte)

        self.central_arena_view_columns        = PandasAugmentedCentralArenaView.central_arena_view_columns


        self.df = PandasAugmentedCentralArenaView.df.copy(deep=True)

        if len(args) == 0:
            return

    def set_data_collection_id(self, id):
        self.auxiliary_variable1_data[0] = id

    def set_time_stamp(self, ts):
        self.auxiliary_variable1_data[1] = ts

    def set_agent_id(self, id):
        self.auxiliary_variable2_data[0] = id

    def set_game_count(self, c):
        self.auxiliary_variable3_data[0] = c

    def set_game_step(self, c):
        self.auxiliary_variable3_data[1] = c

    def set_q_vars(self, crate_destroyed=0, coin_found=0, coin_collected=0, opponent_eliminated=0, killed_opponent=0, got_killed=0, killed_self=0, penalty=0):
        self.q_variable_data = np.array([crate_destroyed, coin_found, coin_collected, opponent_eliminated, killed_opponent, got_killed, killed_self, penalty], dtype=np.byte)

    def set_q_score(self, s):
        self.q_score_data[0] = s

    def set_action(self, a):
        direction = -1
        action    = -1

        switcher = {
            'WAIT' : (-1, -1),
            'UP'   : ( 0, -1),
            'LEFT' : ( 1, -1),
            'DOWN' : ( 2, -1),
            'RIGHT': ( 3, -1),
            'BOMB' : ( -1, 1),
        }

        direction, action = switcher[a]
        self.direction_variable_data[0] = direction
        self.direction_variable_data[1] = action

    def set_survived(self, has_survived):
        self.target_variable_data[0] = has_survived

    def set_won(self, has_won):
        self.target_variable_data[1] = has_won

    def state_from_data(self):
        self.x_new = self.central_arena_view_offset_data[0]
        self.y_new = self.central_arena_view_offset_data[1]
        # print('state_from_data: {}, {}, {}'.format(self.rotation_variable_data, self.rotation_variable_data[0], self.rotation_variable_data[0].count('r')))
        self.rotation = self.rotation_variable_data[0].count('r')
        m = self.rotation_variable_data[0].count('m')
        if m > 0:
            self.mirror = -1

    def copy_from(self, other):
        super().copy_from(other)

        # print('copy_from: other.rotation_variable_data: {},{},{},{} self.rotation_variable_data: {},{},{},{} self.rotation: {}, other.rotation: {}'.format(
        #     other.rotation_variable_data, id(other.rotation_variable_data), type(other.rotation_variable_data), other.rotation_variable_data.dtype,
        #     self.rotation_variable_data, id(self.rotation_variable_data), type(self.rotation_variable_data), self.rotation_variable_data.dtype,
        #     self.rotation, other.rotation))
        self.auxiliary_variable1_data[:]          = other.auxiliary_variable1_data[:]
        self.auxiliary_variable2_data[:]          = other.auxiliary_variable2_data[:]
        self.auxiliary_variable3_data[:]          = other.auxiliary_variable3_data[:]
        self.prediction_variable_data[:]          = other.prediction_variable_data[:]
        self.q_variable_data[:]                   = other.q_variable_data[:]
        self.q_score_data[:]                      = other.q_score_data[:]
        self.target_variable_data[:]              = other.target_variable_data[:]
        self.direction_variable_data[:]           = other.direction_variable_data[:]
        self.rotation_variable_data[:]            = other.rotation_variable_data[:]
        self.central_arena_view_offset_data[:]    = other.central_arena_view_offset_data[:]
        self.state_from_data()

    def rotate_right_(self):
        super().rotate_right_()
        r = 'r' * (self.rotation % 4)
        if self.mirror < 0:
            r += 'm'
        # print('rotate_right_: {},{},{},{},{}'.format(self.rotation, r, self.rotation_variable_data, type(self.rotation_variable_data),self.rotation_variable_data[0]))
        self.rotation_variable_data[0]  = r
        # print('rotate_right_: {},{},{},{},{}'.format(self.rotation, r, self.rotation_variable_data, type(self.rotation_variable_data),self.rotation_variable_data[0]))
        # self.rotation_variable_data  = np.array([r])
        # print('rotate_right_: {},{},{}'.format(self.rotation, r, self.rotation_variable_data))
        self.direction_variable_data[0] = self.direction_rotate_right(self.direction_variable_data[0])

        self.central_arena_view_offset_data[0] = self.x_new
        self.central_arena_view_offset_data[1] = self.y_new

    def mirror_vertical_axis_(self):
        super().mirror_vertical_axis_()
        r = 'r' * (self.rotation % 4)
        if self.mirror < 0:
            r += 'm'
        self.rotation_variable_data[0]  = r
        self.direction_variable_data[0] = self.direction_mirror_vertical_axis(self.direction_variable_data[0])
        self.central_arena_view_offset_data[0] = self.x_new
        self.central_arena_view_offset_data[1] = self.y_new


    def partial_df(self, cols, values):
        d = dict(zip(cols, list(values.reshape(-1, 1))))
        return pd.DataFrame(d)

    def to_df(self):
        ldf_auxiliary_variable1     = self.partial_df(self.auxiliary_variable1_columns, self.auxiliary_variable1_data)
        ldf_auxiliary_variable2     = self.partial_df(self.auxiliary_variable2_columns, self.auxiliary_variable2_data)
        ldf_auxiliary_variable3     = self.partial_df(self.auxiliary_variable3_columns, self.auxiliary_variable3_data)

        r_value = ['r'] * (self.rotation % 4)
        if self.mirror < 0:
            r_value += ['m']
        ldf_rotation                = pd.DataFrame([''.join(r_value)], columns=[DFIDs.ROTATION])

        ldf_prediction_variable     = self.partial_df(self.prediction_variable_columns, self.prediction_variable_data)

        ldf_q_variable              = self.partial_df(self.q_variable_columns, self.q_variable_data)
        ldf_q_score                 = self.partial_df(self.q_score_columns, self.q_score_data)

        ldf_target_variable         = self.partial_df(self.target_variable_columns, self.target_variable_data)

        ldf_direction_variable      = self.partial_df(self.direction_variable_columns, self.direction_variable_data)

        ldf_nearest_other_agent_info = self.partial_df(self.nearest_other_agent_info_columns, self.nearest_other_agent_info_data)
        ldf_nearest_coin_info        = self.partial_df(self.nearest_coin_info_columns       , self.nearest_coin_info_data)
        ldf_nearest_crate_info       = self.partial_df(self.nearest_crate_info_columns     , self.nearest_crate_info_data)
        ldf_mid_of_map_info          = self.partial_df(self.mid_of_map_info_columns     , self.mid_of_map_info_data)

        self.central_arena_view_offset_data[:] = [self.x_new, self.y_new]
        ldf_central_arena_view_offset = self.partial_df(self.central_arena_view_offset_columns, self.central_arena_view_offset_data)
        ldf_central_arena_view = pd.DataFrame(self.central_arena_view.reshape(-1).reshape(1,-1), columns=self.central_arena_view_columns)

        ldf = pd.concat([ldf_auxiliary_variable1, ldf_auxiliary_variable2, ldf_auxiliary_variable3, ldf_rotation, ldf_prediction_variable, ldf_q_variable, ldf_q_score,
                         ldf_target_variable, ldf_direction_variable, ldf_nearest_other_agent_info, ldf_nearest_coin_info, ldf_nearest_crate_info, ldf_mid_of_map_info,
                         ldf_central_arena_view_offset, ldf_central_arena_view], axis=1)

        return ldf

    @staticmethod
    def from_df(ldf):
        obj = PandasAugmentedCentralArenaView()

        obj.auxiliary_variable1_data[:]      = ldf.iloc[0, :].loc[obj.auxiliary_variable1_columns]
        obj.auxiliary_variable2_data[:]      = ldf.iloc[0, :].loc[obj.auxiliary_variable2_columns]
        obj.auxiliary_variable3_data[:]      = ldf.iloc[0, :].loc[obj.auxiliary_variable3_columns]

        if DFIDs.ROTATION not in list(ldf.columns):
            obj.rotation_variable_data[:] = ''
        else:
            obj.rotation_variable_data[:]        = ldf.iloc[0, :].loc[obj.rotation_variable_columns]

        obj.prediction_variable_data[:]      = ldf.iloc[0, :].loc[obj.prediction_variable_columns]

        obj.q_variable_data[:]               = ldf.iloc[0, :].loc[obj.q_variable_columns]
        obj.q_score_data[:]                  = ldf.iloc[0, :].loc[obj.q_score_columns]

        obj.target_variable_data[:]          = ldf.iloc[0, :].loc[obj.target_variable_columns]
        obj.direction_variable_data[:]       = ldf.iloc[0, :].loc[obj.direction_variable_columns]

        rhs = ldf.iloc[0, :].loc[obj.nearest_other_agent_info_columns]
        obj.nearest_other_agent_info_data[:] = ldf.iloc[0, :].loc[obj.nearest_other_agent_info_columns]
        lhs = obj.nearest_other_agent_info_data[:]
        obj.nearest_coin_info_data[:]        = ldf.iloc[0, :].loc[obj.nearest_coin_info_columns]
        obj.nearest_crate_info_data[:]       = ldf.iloc[0, :].loc[obj.nearest_crate_info_columns]
        obj.mid_of_map_info_data[:]          = ldf.iloc[0, :].loc[obj.mid_of_map_info_columns]

        obj.central_arena_view_offset_data[:]      = ldf.iloc[0, :].loc[obj.central_arena_view_offset_columns]

        lds_central_arena_view               = ldf.iloc[0, :].loc[obj.central_arena_view_columns]
        obj.central_arena_view[:,:]          = lds_central_arena_view.values.reshape(-1, obj.central_arena_view.shape[1])

        obj.state_from_data()

        return obj

PACAV = PandasAugmentedCentralArenaView()

class TransformationCache():
    def __init__(self):
        self.cache = {}

    def get_columns_and_breadth(self, cols):
        av_columns = [c for c in cols if c.startswith('cav_')]

        breadth = int(np.sqrt(len(av_columns)))
        if breadth * breadth != len(av_columns):
            raise Exception('The given columns do not generate a square! {}, {}, {}'.format(len(cols), av_columns, cols))

        if breadth % 2 != 1:
            raise Exception('Only squares of uneven side length are allowed')

        size = breadth

        delta = size // 2
        x_min = PACAV.x_center - delta
        y_min = PACAV.y_center - delta

        transformation_fields = [DFIDs.CENTRAL_ARENA_VIEW_FORMAT.format(x, y) for y in range(y_min, y_min + size) for x in range(x_min, x_min + size) ]
        return transformation_fields, breadth

    def get_columns(self, cols):
        return self.get_columns_and_breadth(cols)[0]

    def get_breadth(self, cols):
        return self.get_columns_and_breadth(cols)[1]

    def get_transformation_from_columns(self, cols):
        cols, size = self.get_columns_and_breadth(cols)
        return self.get_transformation(size)

    def get_transformation(self, size):
        if size == 0: # return all columns
            return PACAV.central_arena_view_columns
        if size % 2 != 1:
            raise Exception('only uneven size transformations are allowed')

        transformation_fields = self.cache.get(size, None)
        if transformation_fields is not None:
            return transformation_fields

        delta = size // 2
        x_min = PACAV.x_center - delta
        y_min = PACAV.y_center - delta

        transformation_fields = [DFIDs.CENTRAL_ARENA_VIEW_FORMAT.format(x, y) for y in range(y_min, y_min + size) for x in range(x_min, x_min + size) ]

        self.cache[size] = transformation_fields
        return transformation_fields

TC = TransformationCache()

@numba.jit(nopython=True)
def discount(s, r, base_value=0.0, discount_rate=0.9):
    v = base_value
    if s[len(s)-1] < 0:
        v = 0.0
    for i in range(len(s)-1, -1, -1):
        r[i] = v * discount_rate + s[i]
        v = r[i]



# t5_0, t5_1
class FeatureSelectionTransformation0():

    direction_action_switcher = {
         ( 0,  0) : (0,0,0,0,0,0),
         (-1, -1) : (1,0,0,0,0,0),
         ( 0, -1) : (0,1,0,0,0,0),
         ( 1, -1) : (0,0,1,0,0,0),
         ( 2, -1) : (0,0,0,1,0,0),
         ( 3, -1) : (0,0,0,0,1,0),
         (-1,  1) : (0,0,0,0,0,1),
    }

    direction_switcher = {
        -1: (0, 0),
         0: (0, -1),
         1: (-1, 0),
         2: (0, 1),
         3: (1, 0)
    }

    # w_step_survived       = 1
    w_crate_destroyed     = 1
    w_coin_found          = 1
    w_coin_collected      = 5
    w_opponent_eliminated = 0
    w_killed_opponent     = 10
    w_got_killed          = 0
    w_wasted_move         = 0
    w_wasted_move_qq      = PACAV.bomb_tmax * 1.0 # 3.0

    def __init__(self, in_df, size=5, discount_rate=0.90):
        self.in_df = in_df.copy()

        if DFIDs.ROTATION not in list(self.in_df.columns):
            self.in_df[DFIDs.ROTATION] = ''
            self.in_df = self.in_df[PACAV.df_columns]

        self.size  = size
        #         self.df_columns = self.auxiliary_variable1_columns + self.auxiliary_variable2_columns + self.auxiliary_variable3_columns + self.q_variable_columns + self.q_score_columns +\
        #                           self.target_variable_columns + self.direction_variable_columns + self.nearest_other_agent_info_columns + self.nearest_coin_info_columns + self.nearest_crate_info_columns +
        #                           self.mid_of_map_info_columns + self.central_arena_view_columns
        self.discount_rate = discount_rate

        self.reverse_dmap = dict([(value, key) for key, value in PACAV.dmap.items()])

        # FeatureSelectionTransformation0.w_wasted_move_qq = self.w_wasted_move_qq

    def w_wasted_move_qq_(self):
        r = 1.0/(1.0 - self.discount_rate)
        r = max(0.0, r - 2 * self.w_step_survived())
        return r

    def w_step_survived(self):
        target = 3
        # factor = 0.03
        r = 1.0/(1.0 - self.discount_rate)
        # target = factor * r

        # step = 0.3
        # r    = 10
        # target = step * r
        # step = factor * r
        #
        # target = factor * r^2
        # step = target / r

        step = target / r

        return step

    def select(self, columns):
        #print(list(self.in_df.columns))
        return self.in_df[columns]

    def calculate_qq(self, q, qq_base=0.0, discount_rate=None):
        if discount_rate is None:
            discount_rate = self.discount_rate

        # if qq_base is None:
        #     qq_base = 1.0 / (1.0 - discount_rate)

        # qq = q.copy()
        qq = pd.Series(index=q.index, dtype=np.float32)

        discount(q.values, qq.values, base_value=qq_base, discount_rate=discount_rate)
        return qq

    def calculate_qqs(self, qs):
        qqs = np.full(len(qs), 0.99, dtype=np.float32)
        qqs[0] = 1.0
        qqs = np.cumprod(qqs)[::-1]
        qqs = qqs * qs.iloc[-1,0]

        # qqs = self.calculate_qq(qs.iloc[:,0], qq_base=0.0, discount_rate=0.999)
        r = qs.copy()
        r[DFIDs.QQ_SCORE] = qqs
        return r

    def calculate_q(self, ldf):
        crate_destroyed     = ldf[DFIDs.Q_CRATE_DESTROYED]
        coin_found          = ldf[DFIDs.Q_COIN_FOUND]
        coin_collected      = ldf[DFIDs.Q_COIN_COLLECTED]
        opponent_eliminated = ldf[DFIDs.Q_OPPONENT_ELIMINATED]
        killed_opponent     = ldf[DFIDs.Q_KILLED_OPPONENT]
        got_killed          = ldf[DFIDs.Q_GOT_KILLED]
        killed_self         = ldf[DFIDs.Q_KILLED_SELF]
        # XXX on purpose not inlcuding Q_PENALTY

        # dangerous, because the series below are np.byte!
        w_step_survived       = self.w_step_survived() # FeatureSelectionTransformation0.w_step_survived
        w_crate_destroyed     = FeatureSelectionTransformation0.w_crate_destroyed
        w_coin_found          = FeatureSelectionTransformation0.w_coin_found
        w_coin_collected      = FeatureSelectionTransformation0.w_coin_collected
        w_opponent_eliminated = FeatureSelectionTransformation0.w_opponent_eliminated
        w_killed_opponent     = FeatureSelectionTransformation0.w_killed_opponent
        # w_got_killed          = FeatureSelectionTransformation0.w_got_killed
        # w_got_killed          = -0.6 # -3.0 * w_step_survived
        w_got_killed          = -3.0 * w_step_survived
        # w_killed_self         = -1.5 # -1.0 * (PACAV.bomb_tmax + 2) * w_step_survived
        w_killed_self         = -1.0 * (PACAV.bomb_tmax + 2) * w_step_survived

        idx = ldf.index
        # if got_killed / killed_self is not the last frame then something is wrong
        got_killed_p = got_killed > 0
        killed_self_p = killed_self > 0
        idx_got_killed = got_killed_p | killed_self_p
        if np.sum(idx_got_killed) > 1:
            raise Exception('Surprise: len(idx_got_killed) > 1: {}'.format(idx_got_killed[idx_got_killed]))

        if np.any(idx_got_killed) and idx_got_killed[idx_got_killed].index != idx[-1]:
            raise Exception('Surprise: idx_got_killed does not mark the last line in the dataframe {}, {}'.format(idx_got_killed[idx_got_killed].index, idx.iloc[-1].index))

        # print(ldf.columns)
        idx_wait = (ldf['A'] ==- 1) & (ldf['D'] ==-1)
        # print(idx_wait)
        ds_step_survived = pd.Series(index=ldf.index,dtype=np.float32)
        ds_step_survived.iloc[:] = w_step_survived
        ds_step_survived[idx_wait] = 0

        q = pd.Series(index=idx, name=DFIDs.Q, dtype=np.float32)
        q.loc[got_killed_p]    = w_got_killed
        q.loc[killed_self_p]   = w_killed_self
        q.loc[~idx_got_killed] = ds_step_survived + w_crate_destroyed * crate_destroyed[~idx_got_killed] + w_coin_found * coin_found[~idx_got_killed] + w_coin_collected * coin_collected[~idx_got_killed] + \
                                 w_opponent_eliminated * opponent_eliminated[~idx_got_killed] + w_killed_opponent * killed_opponent[~idx_got_killed]

        qq = self.calculate_qq(q)

        # log.debug('calculate_q: w_step_survived: {}, w_killed_self: {}, w_got_killed: {}, w_crate_destroyed: {}, w_coin_found: {}, w_coin_collected: {}, w_opponent_eliminated: {}, w_killed_opponent: {}, '.format(w_step_survived, w_killed_self, w_got_killed, w_crate_destroyed, w_coin_found, w_coin_collected, w_opponent_eliminated, w_killed_opponent))

        r = q.to_frame()
        r[DFIDs.QQ] = qq

        return r

    def calculate_one_hot_action(self, direction_action_df):
        # valid combinations
        # switcher = {
        #     'WAIT' : (-1, -1),
        #     'UP'   : ( 0, -1),
        #     'LEFT' : ( 1, -1),
        #     'DOWN' : ( 2, -1),
        #     'RIGHT': ( 3, -1),
        #     'BOMB' : ( -1, 1),
        # }

        # switcher = {
        #      (-1, -1) : (1,0,0,0,0,0),
        #      ( 0, -1) : (0,1,0,0,0,0),
        #      ( 1, -1) : (0,0,1,0,0,0),
        #      ( 2, -1) : (0,0,0,1,0,0),
        #      ( 3, -1) : (0,0,0,0,1,0),
        #      (-1,  1) : (0,0,0,0,0,1),
        # }

        rdf = pd.DataFrame(np.zeros((len(direction_action_df), 6)).astype(np.byte), index=direction_action_df.index, columns=DFIDs.A_ONE_HOT)
        da_df = direction_action_df
        a_wait_p  = (da_df[DFIDs.DIRECTION] == -1) & (da_df[DFIDs.ACTION] == -1)
        a_up_p    = (da_df[DFIDs.DIRECTION] ==  0) & (da_df[DFIDs.ACTION] == -1)
        a_left_p  = (da_df[DFIDs.DIRECTION] ==  1) & (da_df[DFIDs.ACTION] == -1)
        a_down_p  = (da_df[DFIDs.DIRECTION] ==  2) & (da_df[DFIDs.ACTION] == -1)
        a_right_p = (da_df[DFIDs.DIRECTION] ==  3) & (da_df[DFIDs.ACTION] == -1)
        a_bomb_p  = (da_df[DFIDs.DIRECTION] == -1) & (da_df[DFIDs.ACTION] ==  1)

        rdf.loc[a_wait_p , DFIDs.A_WAIT]  = 1
        rdf.loc[a_up_p   , DFIDs.A_UP]    = 1
        rdf.loc[a_left_p , DFIDs.A_LEFT]  = 1
        rdf.loc[a_down_p , DFIDs.A_DOWN]  = 1
        rdf.loc[a_right_p, DFIDs.A_RIGHT] = 1
        rdf.loc[a_bomb_p , DFIDs.A_BOMB]  = 1

        return rdf


    def calculate_direction(self, d):
        x = pd.Series(0,index=d.index,dtype=np.byte)
        y = pd.Series(0,index=d.index,dtype=np.byte)

        s = d.iloc[:,0]
        y[s == 0] = -1
        x[s == 1] = -1
        y[s == 2] = +1
        x[s == 3] = +1
        return pd.DataFrame({'{}x'.format(s.name) : x, '{}y'.format(s.name) : y})

    def calculate_nearest_object_data(self, cols):
        dcol = [cols[0]]
        ocols = cols[1:]
        odf = self.select(ocols)
        ddf = self.select(dcol)
        rdf = self.calculate_direction(ddf)
        return pd.concat([rdf, odf], axis=1)

    def calculate_ttl(self):
        idx = self.in_df.index
        rds = None
        if len(self.in_df) >= s.max_steps:
            ttl = s.max_steps
            rds = pd.Series(ttl,index=idx)
        else:
            l = len(idx) - 1
            rds = pd.Series(np.arange(l, -1, -1), index=idx)
        return pd.DataFrame({DFIDs.TTL: rds})

    def in_game_transform_calculate_nearest_object_data(self, data):
        dx, dy = FeatureSelectionTransformation0.direction_switcher[data[0]]
        info = np.zeros(data.shape[0] + 1, dtype=np.byte)
        info[0] = dx
        info[1] = dy
        info[2:] = data[1:]
        return info

    def in_game_transform(self, av):
        # dummy = PandasAugmentedCentralArenaView()

        direction = av.direction_variable_data[0]
        action    = av.direction_variable_data[1]
        one_hot_action = np.array(FeatureSelectionTransformation0.direction_action_switcher[(direction, action)], dtype=np.byte)
        nearest_other_agent_info = self.in_game_transform_calculate_nearest_object_data(av.nearest_other_agent_info_data)
        nearest_coin_info = self.in_game_transform_calculate_nearest_object_data(av.nearest_coin_info_data)
        nearest_crate_info = self.in_game_transform_calculate_nearest_object_data(av.nearest_crate_info_data)
        mid_of_map_info = self.in_game_transform_calculate_nearest_object_data(av.mid_of_map_info_data)

        delta = self.size // 2
        x_min = PACAV.x_center - delta
        y_min = PACAV.y_center - delta

        central_arena_view = av.central_arena_view[y_min:y_min+self.size,x_min:x_min+self.size].reshape(-1)

        r = np.concatenate([one_hot_action, nearest_other_agent_info, nearest_coin_info, nearest_crate_info, mid_of_map_info, central_arena_view])

        return r

    def transform(self):
        ldf_auxiliary_variable1     = self.select(PACAV.auxiliary_variable1_columns)
        ldf_auxiliary_variable2     = self.select(PACAV.auxiliary_variable2_columns)
        ldf_auxiliary_variable3     = self.select(PACAV.auxiliary_variable3_columns)

        ldf_rotation                = self.select(PACAV.rotation_variable_columns) # pd.DataFrame('', index=self.in_df.index, columns=[DFIDs.ROTATION])

        ldf_q_variable = self.calculate_q(self.select(PACAV.q_variable_columns + ['D', 'A']))

        ldf_q_score    = self.calculate_qqs(self.select(PACAV.q_score_columns))

        ldf_target_variable = self.select(PACAV.target_variable_columns)

        ldf_ttl        = self.calculate_ttl()

        # ldf_direction_variable = self.calculate_direction(self.select([DFIDs.DIRECTION]))
        # ldf_direction_variable[DFIDs.ACTION] = self.select([DFIDs.ACTION]).iloc[:,0]
        ldf_one_hot_action           = self.calculate_one_hot_action(self.select([DFIDs.DIRECTION, DFIDs.ACTION]))

        ldf_nearest_other_agent_info = self.calculate_nearest_object_data(PACAV.nearest_other_agent_info_columns)
        ldf_nearest_coin_info        = self.calculate_nearest_object_data(PACAV.nearest_coin_info_columns)
        ldf_nearest_crate_info       = self.calculate_nearest_object_data(PACAV.nearest_crate_info_columns)
        ldf_mid_of_map_info          = self.calculate_nearest_object_data(PACAV.mid_of_map_info_columns)

        ldf_central_arena_view_offset = self.select(PACAV.central_arena_view_offset_columns)

        ldf_central_arena_view        = self.select(TC.get_transformation(self.size))

        ldf = pd.concat([ldf_auxiliary_variable1, ldf_auxiliary_variable2, ldf_auxiliary_variable3, ldf_rotation, ldf_q_variable, ldf_q_score,
                         ldf_target_variable, ldf_ttl, ldf_one_hot_action, ldf_nearest_other_agent_info, ldf_nearest_coin_info, ldf_nearest_crate_info, ldf_mid_of_map_info,
                         ldf_central_arena_view_offset, ldf_central_arena_view], axis=1)

        return ldf

class LevelPrinter():
    def __init__(self, ldf, strip_if_possible=True):
        av_columns = TC.get_columns(ldf.columns)
        self.ldf = ldf[av_columns]
        self.strip = strip_if_possible and (len(av_columns) >= PACAV.central_arena_view.shape[0] * PACAV.central_arena_view.shape[1])

    def print(self, row_idx=0):
        row = self.ldf.iloc[row_idx,:]

        breadth = TC.get_breadth(row.index)
        # breadth = int(np.sqrt(len(row)))
        # if breadth * breadth != len(row):
        #     raise Exception('The given row is not a square! {}'.format(row.shape))
        view_array = row.values.reshape(-1, breadth)
        self.view_array = view_array

        # print(view_array)
        # coordinates = np.where(view_array == PACAV.code_me)
        # print(coordinates)
        # print(view_array[coordinates[0], coordinates[1]])
        if self.strip:
            # strip rows
            rows = np.all(view_array == -1, axis=1)
            view_array = view_array[~rows,:]
            # strip cols
            cols = np.all(view_array == -1, axis=0)
            view_array = view_array[:,~cols]

        self.view_array = view_array

        char_array = np.full(view_array.shape, '.')

        # set walls
        coordinates = PACAV.is_code(view_array, PACAV.code_wall)
        char_array[coordinates[0], coordinates[1]] = '#'
        # set free spots
        coordinates = PACAV.is_code(view_array, PACAV.code_free)
        char_array[coordinates[0], coordinates[1]] = '.'
        # set crates
        coordinates = PACAV.is_code(view_array, PACAV.code_crate)
        char_array[coordinates[0], coordinates[1]] = '+'
        # set players
        coordinates = PACAV.is_code(view_array, PACAV.code_other_agent)
        char_array[coordinates[0], coordinates[1]] = '@'
        # if self.strip:
        #     char_array[coordinates[0], coordinates[1]] = 'M'
        # else:
        #     char_array[coordinates[0], coordinates[1]] = '@'

        # set coins
        coordinates = PACAV.is_code(view_array, PACAV.code_coin)
        char_array[coordinates[0], coordinates[1]] = '$'

        # set bombs
        coordinates = PACAV.is_code(view_array, PACAV.code_bomb)
        char_array[coordinates[0], coordinates[1]] = 'B'

        # set explosion
        coordinates = np.where(view_array >= PACAV.code_bomb_explosion)
        char_array[coordinates[0], coordinates[1]] = 'E'

        # set me
        x = view_array.shape[1] // 2
        y = view_array.shape[0] // 2
        if char_array[y, x] == '.':
            char_array[y, x] = 'M'

        for ri in range(char_array.shape[0]):
            print(''.join(char_array[ri,:]))

# TODO:
# - take care of origin ox, oy in batch rotations

@numba.jit(nopython=True, parallel=True)
def batch_rotate(level_stack, result_level_stack):
    for z in numba.prange(level_stack.shape[0]):
        for i in range(level_stack.shape[1]):
            result_level_stack[z, i, :] = level_stack[z, ::-1, i]

class PostProcessGame():
    def __init__(self, in_file_names, out_file_name, size=5, verify=False, print_p=False):
        self.in_file_names = in_file_names
        self.out_file_name = out_file_name
        self.size    = size
        self.verify  = verify
        self.print_p = print_p

        # id = '1550494638'
        # file_pattern = '{}-simple_agent_{{}}.h5'.format(id)
        # print(file_pattern)
        # file_name = './hdf5_training_data/{}-simple_agent_0.h5'.format(id)
        # store = pd.HDFStore(file_name)
        # # pd.read_hdf('file_name', 'df')
        # df = store['simple_agent_0_0']
        # store.close()

        self.store0 = pd.HDFStore(in_file_names[0])
        self.store0keys = self.store0.keys()
        self.store1 = pd.HDFStore(in_file_names[1])
        self.store1keys = self.store1.keys()
        self.store2 = pd.HDFStore(in_file_names[2])
        self.store2keys = self.store2.keys()
        self.store3 = pd.HDFStore(in_file_names[3])
        self.store3keys = self.store3.keys()

        l = len(self.store0keys)
        l_ = len(self.store1keys)
        if l_ != l:
            raise Exception('store1 has different number of games than store0: {} vs. {}'.format(l_, l))
        l_ = len(self.store2keys)
        if l_ != l:
            raise Exception('store2 has different number of games than store0: {} vs. {}'.format(l_, l))
        l_ = len(self.store3keys)
        if l_ != l:
            raise Exception('store3 has different number of games than store0: {} vs. {}'.format(l_, l))

        self.l = l

        self.transform_game_timings = []
        self.feature_transform_timings = []

    def find_winner(self, player_dfs):
        scores_ = [pdf.iloc[-1]['QSC'] for pdf in player_dfs]
        scores = np.array(scores_)
        winner_idx = np.argmax(scores)
        player_dfs[winner_idx].loc[:,'W'] = True
        # print('winner_idx: {}'.format(winner_idx))

    def feature_transform(self, player_dfs):
        rdfs = []

        for pdf in player_dfs:
            time1 = time.time()
            ft = FeatureSelectionTransformation0(pdf, size=self.size)
            r  = ft.transform()
            time2 = time.time()
            self.feature_transform_timings += [(time2-time1)*1000.0]
            rdfs += [r]

        return rdfs

    def batch_rotate_one_hot_action(self, tpdf):
        a_up_p    = tpdf[DFIDs.A_UP]    == 1
        a_left_p  = tpdf[DFIDs.A_LEFT]  == 1
        a_down_p  = tpdf[DFIDs.A_DOWN]  == 1
        a_right_p = tpdf[DFIDs.A_RIGHT] == 1
        tpdf.loc[a_up_p   , [DFIDs.A_UP   , DFIDs.A_RIGHT]] = (0, 1) # set the old value to zero and the new value to one
        tpdf.loc[a_left_p , [DFIDs.A_LEFT , DFIDs.A_UP]]    = (0, 1)
        tpdf.loc[a_down_p , [DFIDs.A_DOWN , DFIDs.A_LEFT]]  = (0, 1)
        tpdf.loc[a_right_p, [DFIDs.A_RIGHT, DFIDs.A_DOWN]]  = (0, 1)


    def batch_rotate_direction(self, tpdf, direction_columns):
        dx, dy    = direction_columns
        idx_up    = (tpdf[dx] ==  0) & (tpdf[dy] == -1)
        idx_left  = (tpdf[dx] == -1) & (tpdf[dy] ==  0)
        idx_down  = (tpdf[dx] ==  0) & (tpdf[dy] ==  1)
        idx_right = (tpdf[dx] == +1) & (tpdf[dy] ==  0)
        tpdf.loc[idx_up,direction_columns]    = [+1,  0]
        tpdf.loc[idx_left,direction_columns]  = [ 0, -1]
        tpdf.loc[idx_down,direction_columns]  = [-1,  0]
        tpdf.loc[idx_right,direction_columns] = [ 0, +1]

    def direction_columns(self, direction_group):
        return direction_group[0] + 'x', direction_group[0] + 'y'

    # old_x_new = x_new
    # old_y_new = y_new
    #
    # lower_left_x = old_x_new
    # lower_left_y = old_y_new + self.delta_y - 1
    #
    # x_new = 28 - lower_left_y
    # y_new = lower_left_x

    def batch_rotate_offset(self, tpdf):
        lds_old_x_new = tpdf[DFIDs.CENTRAL_ARENA_VIEW_OFFSET_X].copy()
        lds_old_y_new = tpdf[DFIDs.CENTRAL_ARENA_VIEW_OFFSET_Y].copy()
        tpdf[DFIDs.CENTRAL_ARENA_VIEW_OFFSET_X] = 28 - (lds_old_y_new + PACAV.delta_y - 1)
        tpdf[DFIDs.CENTRAL_ARENA_VIEW_OFFSET_Y] = lds_old_x_new

    def batch_rotate_arena(self, rdf, av_columns):
        breadth = int(np.sqrt(len(av_columns)))

        cav_df = rdf[av_columns]
        row_count = len(cav_df)
        level_stack = cav_df.values.reshape(row_count, -1, breadth)
        result_level_stack = np.full(level_stack.shape, 0, np.byte)
        batch_rotate(level_stack, result_level_stack)
        # for z in range(level_stack.shape[0]):
        #     for i in range(level_stack.shape[1]):
        #         result_level_stack[z, i,:] = level_stack[z,::-1,i]

        rdf.loc[:,av_columns] = result_level_stack.reshape(row_count,-1)

    def batch_rotate_verify(self, in_df, out_df, rotations=1):
        test_indices = np.random.choice(np.arange(len(in_df)), 10, replace=False)
        filter_columns = [DFIDs.QQ, DFIDs.QQ_SCORE, DFIDs.TTL, DFIDs.W]
        compare_columns = [c for c in out_df.columns if c not in filter_columns]
        for idx in test_indices:
            in_df_row_df = in_df.iloc[[idx],:]
            lpacav = PandasAugmentedCentralArenaView.from_df(in_df_row_df)
            if self.print_p:
                print('in_df_row_df:')
                ft = FeatureSelectionTransformation0(lpacav.to_df(), size=self.size)
                in_df_r  = ft.transform()
                print(in_df_r[compare_columns])
            rr = lpacav
            # print('batch_rotate_verify, lpacav rotate_right: {}, {}, {}'.format(-1, rr.rotation, rr.rotation_variable_data))
            for i in range(rotations):
                rr = rr.rotate_right()
                # print(rr.target_variable_data)
                # print('batch_rotate_verify, lpacav rotate_right: {}, {}, {}'.format(i, rr.rotation, rr.rotation_variable_data))
            ft = FeatureSelectionTransformation0(rr.to_df(), size=self.size)
            r  = ft.transform()
            if self.print_p:
                print('r:')
                print(r[compare_columns])
                print('out_df.iloc[[idx]]:')
                tmp_df = out_df.iloc[[idx]].copy().reset_index(drop=True)
                tmp_df['S'] = tmp_df['S'].astype(np.byte)
                tmp_df['W'] = tmp_df['W'].astype(np.byte)
                print(tmp_df[compare_columns])

            lds1 = out_df.iloc[idx][compare_columns]
            lds2 = r[compare_columns].iloc[0,:]
            # print('{}, {}'.format(type(lds1), type(lds2)))
            lds =  lds1 == lds2

            # print(lds[~lds])
            if np.any(~lds):
                lds_failed = lds[~lds]
                lindf = in_df_row_df[list(lds_failed.index)]
                raise Exception('batch_rotate_verify failed:\nproblematic field:\n{}\nlds1:\n{}\nlds2:\n{}\nin_df:\n{}'.format(lds_failed, lds1[~lds], lds2[~lds], lindf))


    def batch_rotate_right(self, tpdf, av_columns):
        rpdf = tpdf.copy()

        rpdf[DFIDs.ROTATION] = rpdf[DFIDs.ROTATION] + 'r'

        self.batch_rotate_one_hot_action(rpdf)

        self.batch_rotate_direction(rpdf, self.direction_columns(PACAV.nearest_other_agent_info_columns))
        self.batch_rotate_direction(rpdf, self.direction_columns(PACAV.nearest_coin_info_columns))
        self.batch_rotate_direction(rpdf, self.direction_columns(PACAV.nearest_crate_info_columns))
        self.batch_rotate_direction(rpdf, self.direction_columns(PACAV.mid_of_map_info_columns))

        self.batch_rotate_offset(rpdf)
        self.batch_rotate_arena(rpdf, av_columns)

        return rpdf

    def batch_mirror_one_hot_action(self, tpdf):
        a_left_p  = tpdf[DFIDs.A_LEFT]  == 1
        a_right_p = tpdf[DFIDs.A_RIGHT] == 1
        tpdf.loc[a_left_p , [DFIDs.A_LEFT , DFIDs.A_RIGHT]] = (0, 1)
        tpdf.loc[a_right_p, [DFIDs.A_RIGHT, DFIDs.A_LEFT]]  = (0, 1)

    def batch_mirror_direction(self, tpdf, direction_columns):
        dx, dy    = direction_columns
        # idx_up    = (tpdf[dx] ==  0) & (tpdf[dy] == -1)
        idx_left  = (tpdf[dx] == -1) & (tpdf[dy] ==  0)
        # idx_down  = (tpdf[dx] ==  0) & (tpdf[dy] ==  1)
        idx_right = (tpdf[dx] == +1) & (tpdf[dy] ==  0)
        # tpdf.loc[idx_up,direction_columns]    = [+1,  0]
        tpdf.loc[idx_left,direction_columns]  = [ +1, 0]
        # tpdf.loc[idx_down,direction_columns]  = [-1,  0]
        tpdf.loc[idx_right,direction_columns] = [ -1, 0]

    def batch_mirror_offset(self, tpdf):
        lds_old_x_new = tpdf[DFIDs.CENTRAL_ARENA_VIEW_OFFSET_X].copy()
        tpdf[DFIDs.CENTRAL_ARENA_VIEW_OFFSET_X] = 28 - (lds_old_x_new + PACAV.delta_x - 1)


    def batch_mirror_arena(self, rdf, av_columns):
        breadth = int(np.sqrt(len(av_columns)))

        # r = self.central_arena_view[:,::-1].copy()

        cav_df = rdf[av_columns]
        row_count = len(cav_df)
        level_stack = cav_df.values.reshape(row_count, -1, breadth)
        result_level_stack = level_stack[:,:,::-1].copy()

        rdf.loc[:,av_columns] = result_level_stack.reshape(row_count,-1)

    def batch_mirror_vertical(self, tpdf, av_columns):
        rpdf = tpdf.copy()

        rpdf[DFIDs.ROTATION] = rpdf[DFIDs.ROTATION] + 'm'

        self.batch_mirror_one_hot_action(rpdf)

        self.batch_mirror_direction(rpdf, self.direction_columns(PACAV.nearest_other_agent_info_columns))
        self.batch_mirror_direction(rpdf, self.direction_columns(PACAV.nearest_coin_info_columns))
        self.batch_mirror_direction(rpdf, self.direction_columns(PACAV.nearest_crate_info_columns))
        self.batch_mirror_direction(rpdf, self.direction_columns(PACAV.mid_of_map_info_columns))

        self.batch_mirror_offset(rpdf)
        self.batch_mirror_arena(rpdf, av_columns)

        return rpdf

    def batch_mirror_verify(self, in_df, out_df):
        test_indices = np.random.choice(np.arange(len(in_df)), 3, replace=False)
        filter_columns = [DFIDs.QQ, DFIDs.QQ_SCORE, DFIDs.TTL, DFIDs.W]
        compare_columns = [c for c in out_df.columns if c not in filter_columns]
        for idx in test_indices:
            in_df_row_df = in_df.iloc[[idx],:]
            lpacav = PandasAugmentedCentralArenaView.from_df(in_df_row_df)
            if self.print_p:
                print('in_df_row_df:')
                ft = FeatureSelectionTransformation0(lpacav.to_df(), size=self.size)
                in_df_r  = ft.transform()
                print(in_df_r[compare_columns])
            rr = lpacav.mirror_vertical_axis()
            ft = FeatureSelectionTransformation0(rr.to_df(), size=self.size)
            r  = ft.transform()
            if self.print_p:
                print('r:')
                print(r[compare_columns])
                print('out_df.iloc[[idx]]:')
                tmp_df = out_df.iloc[[idx]].copy().reset_index(drop=True)
                tmp_df['S'] = tmp_df['S'].astype(np.byte)
                tmp_df['W'] = tmp_df['W'].astype(np.byte)
                print(tmp_df[compare_columns])

            lds1 = out_df.iloc[idx][compare_columns]
            lds2 = r[compare_columns].iloc[0,:]
            # print('{}, {}'.format(type(lds1), type(lds2)))
            lds =  lds1 == lds2

            # print(lds[~lds])
            if np.any(~lds):
                lds_failed = lds[~lds]
                lindf = in_df_row_df[list(lds_failed.index)]
                raise Exception('batch_mirror_verify failed:\nproblematic field:\n{}\nlds1:\n{}\nlds2:\n{}\nin_df:\n{}'.format(lds_failed, lds1[~lds], lds2[~lds], lindf))


    def transform_game(self, idx):
        time1 = time.time()

        p0df = self.store0[self.store0keys[idx]].copy()
        p1df = self.store1[self.store1keys[idx]].copy()
        p2df = self.store2[self.store2keys[idx]].copy()
        p3df = self.store3[self.store3keys[idx]].copy()
        player_dfs = [p0df, p1df, p2df, p3df]
        time2 = time.time()

        self.find_winner(player_dfs)
        time3 = time.time()

        transformed_player_dfs = pd.concat(self.feature_transform(player_dfs), axis=0)
        time4 = time.time()
        self.transform_game_timings += [((time2-time1)*1000.0, (time3-time2)*1000.0, (time4-time3)*1000.0, (time4-time1)*1000.0)]
        return transformed_player_dfs

    def transformed_game_df_rotate_and_mirror(self, transformed_player_dfs):
        av_columns = TC.get_columns(transformed_player_dfs.columns)

        rpdf1 = self.batch_rotate_right(transformed_player_dfs, av_columns) # creates a copy

        rpdf2 = self.batch_rotate_right(rpdf1, av_columns) # creates a copy

        rpdf3 = self.batch_rotate_right(rpdf2, av_columns) # creates a copy

        rpdf = pd.concat([transformed_player_dfs, rpdf1, rpdf2, rpdf3])

        mpdf = self.batch_mirror_vertical(rpdf, av_columns)

        rpdf = pd.concat([rpdf, mpdf], axis=0)

        return rpdf

    # transformed_games_list = [self.transform_game(idx) for idx in range(self.l)]

    # self.find_winner(player_dfs)
    # transformed_player_dfs = pd.concat(self.feature_transform(player_dfs), axis=0)

    def process_game(self, idx):
        #
        # self.find_winner(player_dfs)
        #
        # transformed_player_dfs = pd.concat(self.feature_transform(player_dfs), axis=0)

        transformed_player_dfs = self.transform_game(idx)

        av_columns = TC.get_columns(transformed_player_dfs.columns)

        # print(self.transformed_player_dfs.head())
        # print(len(transformed_player_dfs), len(p0df), len(p1df), len(p2df), len(p3df))
        rpdf1 = self.batch_rotate_right(transformed_player_dfs, av_columns) # creates a copy
        # print(rpdf.head())

        p0df = None
        p1df = None
        p2df = None
        p3df = None
        player_dfs = [p0df, p1df, p2df, p3df]
        if self.verify:
            p0df = self.store0[self.store0keys[idx]]
            p1df = self.store1[self.store1keys[idx]]
            p2df = self.store2[self.store2keys[idx]]
            p3df = self.store3[self.store3keys[idx]]
            player_dfs = [p0df, p1df, p2df, p3df]

        if self.verify:
            if self.print_p:
                print('batch_rotate_verify: r1')
            self.batch_rotate_verify(pd.concat(player_dfs, axis=0), rpdf1)

        rpdf2 = self.batch_rotate_right(rpdf1, av_columns) # creates a copy
        if self.verify:
            if self.print_p:
                print('batch_rotate_verify: r2')
            self.batch_rotate_verify(pd.concat(player_dfs, axis=0), rpdf2, rotations=2)

        rpdf3 = self.batch_rotate_right(rpdf2, av_columns) # creates a copy
        if self.verify:
            if self.print_p:
                print('batch_rotate_verify: r3')
            self.batch_rotate_verify(pd.concat(player_dfs, axis=0), rpdf3, rotations=3)

        rpdf = pd.concat([transformed_player_dfs, rpdf1, rpdf2, rpdf3])

        mpdf = self.batch_mirror_vertical(rpdf, av_columns)

        if self.verify:
            if self.print_p:
                print('batch_mirror_verify')
            mpdf1 = self.batch_mirror_vertical(transformed_player_dfs, av_columns)
            self.batch_mirror_verify(pd.concat(player_dfs, axis=0), mpdf1)

        rpdf = pd.concat([rpdf, mpdf], axis=0)

        # print(len(rpdf))

        return rpdf

    def process(self):
        time1 = time.time()
        transformed_games_list = [self.transform_game(idx) for idx in range(self.l)]
        time2 = time.time()
        transformed_games_df = pd.concat(transformed_games_list, axis=0)
        time3 = time.time()
        del transformed_games_list
        time4 = time.time()

        r = self.transformed_game_df_rotate_and_mirror(transformed_games_df)
        time5 = time.time()
        r.reset_index(drop=True, inplace=True)
        time6 = time.time()
        with pd.HDFStore(self.out_file_name, mode='w') as s:
            s.put('df', r)
        time7 = time.time()
        tgts1to2list = [t[0] for t in self.transform_game_timings]
        tgts2to3list = [t[1] for t in self.transform_game_timings]
        tgts3to4list = [t[2] for t in self.transform_game_timings]
        tgts1to4list = [t[3] for t in self.transform_game_timings]
        tgts1to2mean = np.mean(tgts1to2list)
        tgts1to2sum  = np.sum(tgts1to2list)
        tgts2to3mean = np.mean(tgts2to3list)
        tgts2to3sum  = np.sum(tgts2to3list)
        tgts3to4mean = np.mean(tgts3to4list)
        tgts3to4sum  = np.sum(tgts3to4list)
        tgts1to4mean = np.mean(tgts1to4list)
        tgts1to4sum  = np.sum(tgts1to4list)
        tgtstimings = dict(tgts1to2mean=tgts1to2mean, tgts1to2sum=tgts1to2sum, tgts2to3mean=tgts2to3mean, tgts2to3sum=tgts2to3sum, tgts3to4mean=tgts3to4mean, tgts3to4sum=tgts3to4sum, tgts1to4mean=tgts1to4mean, tgts1to4sum=tgts1to4sum )
        self.time_info = ((time2-time1)*1000.0, (time3-time2)*1000.0, (time4-time3)*1000.0, (time5-time4)*1000.0, (time6-time5)*1000.0, (time7-time6)*1000.0,
                          tgtstimings,
                          np.sum(self.feature_transform_timings), np.mean(self.feature_transform_timings))

    def process_(self):
        with pd.HDFStore(self.out_file_name, mode='w') as s:
            for idx in range(self.l):
                r = self.process_game(idx)
                s.put('df', r, format='table', append=True)
            # self.result_store.put('df', r, format='table', append=True)
            # if idx == 0:
            #     self.result_store.put('df', r, format='table', append=True)
            # else:
            #     self.result_store.append('df', r, format='table')

            s['df'].reset_index(drop=True, inplace=True)

    def __del__(self):
        self.store0.close()
        self.store1.close()
        self.store2.close()
        self.store3.close()

# TODO:
# - add penalty info to Q-values as OR bit pattern
# + for every row add rows that give QQ=-2 for directions that are impossible and would waste moves
# - for every row add rows that give QQ=-10 for waiting on bombs or fields that will explode
# - print win statistics when transforming input files to a file, as "won" and as "total score"
# - add tests / verification of the code
#
# - run.sh -> run.py, learning
# - train baseline model with two fully connected layers
# - integrate trained model
# - run.sh ->
# -  loop
# -   run game
# -   run learning
#
# - extend the data augmentation process for waiting on bombs or waiting on explosion fields
# - remove your own character code from the game representatin. this does not add any value, because you're always located on (14,14), the center.
#   but putting your character code on a field may hide the bomb that you put there yourself, e.g. the model cannot learn how to react on this situation


class AugmentGameDataWithPenaltyMoves():

    def __init__(self, in_df):
        self.in_df = in_df
        self.tmax = PACAV.bomb_tmax
        cols, breadth = TC.get_columns_and_breadth(self.in_df.columns)
        self.cav_columns = cols
        self.breadth     = breadth
        self.lx = PACAV.x_center - self.breadth//2
        self.rx = PACAV.x_center + self.breadth//2 + 1
        self.uy = PACAV.y_center - self.breadth//2
        self.dy = PACAV.y_center + self.breadth//2 + 1

    # self.code_wall           = -1
    # self.code_free           =  0
    # self.code_crate          =  1
    # self.code_coin           =  2
    # self.code_me             =  3
    # self.code_other_agent    =  4
    # self.code_bomb           =  5
    # self.code_bomb_base      = 10
    # self.code_bomb_explosion = 50

    # tmax = s.bomb_timer + 1
    # is_exact  = (np_array == base_code)
    # is_bombed = (np_array >= self.code_bomb_base + base_code * tmax) & (np_array < self.code_bomb_base + (base_code+1) * tmax)
    def idx_for_code(self, field_id, base_code):
        idx = (self.in_df[field_id] == base_code) | \
              ((self.in_df[field_id] >= PACAV.code_bomb_base + base_code * self.tmax) & (self.in_df[field_id] < PACAV.code_bomb_base + (base_code+1) * self.tmax))
        return idx


    def process_direction(self, coordinates, one_hot_encoding):
        x,y = coordinates
        field_id = DFIDs.CENTRAL_ARENA_VIEW_FORMAT.format(x, y)
        idx1 = self.idx_for_code(field_id, PACAV.code_wall) | \
               self.idx_for_code(field_id, PACAV.code_crate)| \
               self.idx_for_code(field_id, PACAV.code_other_agent)| \
               self.idx_for_code(field_id, PACAV.code_bomb)

        idx2 = self.in_df[field_id] >= PACAV.code_bomb_explosion + 1

        idx3 = (self.in_df[field_id] >= PACAV.code_bomb_base) & (self.in_df[field_id] < PACAV.code_bomb_explosion) & ((self.in_df[field_id] - PACAV.code_bomb_base) % self.tmax == 0)

        rdf1 =self.in_df[idx1 & ~idx3].copy()
        rdf1[DFIDs.Q]  = FeatureSelectionTransformation0.w_wasted_move
        rdf1[DFIDs.QQ] = FeatureSelectionTransformation0.w_wasted_move_qq
        rdf1[DFIDs.A_ONE_HOT] = one_hot_encoding

        rdf2 =self.in_df[idx2].copy()
        rdf2[DFIDs.Q]  = FeatureSelectionTransformation0.w_got_killed
        rdf2[DFIDs.QQ] = FeatureSelectionTransformation0.w_got_killed
        rdf2[DFIDs.A_ONE_HOT] = one_hot_encoding

        rdf3 =self.in_df[idx3].copy()
        rdf3[DFIDs.Q]  = FeatureSelectionTransformation0.w_got_killed
        rdf3[DFIDs.QQ] = FeatureSelectionTransformation0.w_got_killed
        rdf3[DFIDs.A_ONE_HOT] = one_hot_encoding

        rdf = pd.concat([rdf1, rdf2, rdf3], axis=0)

        return rdf


    def get_bomb_blast_coords(self, arena, x, y, rng):
        blast_coords = [(x,y)]

        for i in range(1, rng+1):
            if arena[y,x+i] == -1: break
            blast_coords.append((x+i,y))
        for i in range(1, rng+1):
            if arena[y,x-i] == -1: break
            blast_coords.append((x-i,y))
        for i in range(1, rng+1):
            if arena[y+i,x] == -1: break
            blast_coords.append((x,y+i))
        for i in range(1, rng+1):
            if arena[y-i,x] == -1: break
            blast_coords.append((x,y-i))

        return blast_coords

    def calculate_bomb_pattern(self, arena, t):
        rng = int(np.min([self.breadth//2, s.bomb_power]))
        blast_coordinates = self.get_bomb_blast_coords(arena, 14, 14, rng)

        arena[14,14] = PACAV.calc_bomb_field_code(PACAV.code_bomb, self.tmax, t)

        for bc in blast_coordinates:
            x,y = bc
            base_code = arena[y,x]
            if base_code >= PACAV.code_bomb_base: # do not add several times the values
                continue
            arena[y, x] = PACAV.calc_bomb_field_code(base_code, self.tmax, t)

    def cav_fields_to_arena(self, lds_row):
        cav_fields = lds_row[self.cav_columns]
        arena_ = cav_fields.values.reshape(self.breadth, self.breadth)
        arena = np.full((28,28),-1)
        arena[self.lx:self.rx,self.uy:self.dy] = arena_
        return arena

    def arena_to_cav_fields(self, arena):
        arena_ = arena[self.lx:self.rx,self.uy:self.dy]
        lds = pd.Series(arena_.reshape(-1), index=self.cav_columns)
        return lds

    def process_wait_on_bomb(self):

        rdf = pd.DataFrame(columns=self.in_df.columns)

        fraction   = 0.0001
        n          = len(self.in_df)
        fraction_n = max(1,int(n * fraction))
        idx        = np.random.permutation(np.arange(0, n))[:fraction_n]
        in_df      = self.in_df.iloc[idx,:]

        log.debug('process_wait_on_bomb fraction_n: {}'.format(fraction_n))

        for t in range(self.tmax):

            for index, row in in_df.iterrows():
                a = self.cav_fields_to_arena(row)
                self.calculate_bomb_pattern(a, t)
                lds = self.arena_to_cav_fields(a)
                r = row.copy()
                r.loc[lds.index] = lds
                r[DFIDs.Q] = FeatureSelectionTransformation0.w_wasted_move     - (self.tmax - t)
                r[DFIDs.QQ] = FeatureSelectionTransformation0.w_wasted_move_qq - (self.tmax - t)
                r2 = r.copy()
                r[DFIDs.A_ONE_HOT]  = (1, 0, 0, 0, 0, 0)
                r2[DFIDs.A_ONE_HOT] = (0, 0, 0, 0, 0, 1)
                rdf.loc[len(rdf)] = r
                rdf.loc[len(rdf)] = r2

        return rdf

    # A_ONE_HOT             = ['A_WAIT', 'A_UP', 'A_LEFT', 'A_DOWN', 'A_RIGHT', 'A_BOMB']
    def process_(self):
        # move up not possible, e.g. field 14,13 is a -1

        rdf_up    = self.process_direction((14,13), (0,1,0,0,0,0))
        rdf_left  = self.process_direction((13,14), (0,0,1,0,0,0))
        rdf_down  = self.process_direction((14,15), (0,0,0,1,0,0))
        rdf_right = self.process_direction((15,14), (0,0,0,0,1,0))

        rdf_wait_on_bomb = self.process_wait_on_bomb()

        rdf = pd.concat([rdf_up, rdf_left, rdf_down, rdf_right, rdf_wait_on_bomb])
        return rdf

    def process(self):
        rdf1 = self.process_()
        rdf = pd.concat([self.in_df, rdf1], axis=0)
        rdf.sort_values(PACAV.sort_columns, inplace=True)
        rdf.reset_index(drop=True, inplace=True)
        return rdf


class FeatureSelectionTransformationNCHW():

    channels = ['origin', 'walls', 'crates', 'coins', 'agents', 'bombs', 'btimes', 'explosions']

    def __init__(self, bt_df):
        # self.in_df = in_df.copy()
        #
        # self.base_transform = FeatureSelectionTransformation0(self.in_df, size=11)
        self.bt_df = bt_df

    def is_code(self, xin, code):
        return PACAV.is_code_p(xin.loc[dict(channel='origin')].values, code)

    def bomb_time(self, xin):
        xin = xin.loc[dict(channel='origin')].values

        idx1 = xin >= PACAV.code_bomb_base
        idx2 = xin < PACAV.code_bomb_explosion

        t = (xin - PACAV.code_bomb_base) % PACAV.bomb_tmax

        # r = np.where(idx1 & idx2)

        r = np.full(xin.shape, PACAV.bomb_tmax)
        r[idx1 & idx2] = t[idx1 & idx2]
        r = r / PACAV.bomb_tmax

        return r

    def explosion_time(self, xin):
        xin = xin.loc[dict(channel='origin')].values

        idx = xin >= PACAV.code_bomb_explosion

        t = xin - PACAV.code_bomb_explosion + 1

        # r = np.where(idx1 & idx2)

        r = np.zeros(xin.shape)
        r[idx] = t[idx]

        r = r / 2.0

        return r

    def transform(self):
        # bt_df = self.base_transform.transform()
        bt_df = self.bt_df

        # base_columns = [c for c in bt_df.columns if not c.startswith('')]
        transformation_fields, breadth = TC.get_columns_and_breadth(bt_df.columns)
        base_column_length             = len(bt_df.columns) - breadth * breadth
        base_columns                   = bt_df.columns[:base_column_length]
        cav_columns                    = transformation_fields
        subject_index                  = range(len(bt_df))

        base_array = xr.DataArray(bt_df[base_columns].values, dims=['subject', 'base_fields'], coords=[subject_index, list(base_columns)])

        cav_array = xr.DataArray(bt_df[cav_columns].values, dims=['subject', 'xy'], coords=[subject_index, list(cav_columns)])
        x   = range(14 - breadth // 2, 14 + breadth // 2 + 1)
        y   = range(14 - breadth // 2, 14 + breadth // 2 + 1)
        ind = pd.MultiIndex.from_product((y, x), names=('y', 'x'))

        tmp_xds = cav_array.to_dataset(name='levels')
        tmp_xds = tmp_xds.assign(xy=ind).unstack('xy')
        cav_array = tmp_xds['levels']
        del tmp_xds
        # idx     = pd.Index(range(breadth * breadth))

        channels = FeatureSelectionTransformationNCHW.channels
        l = len(channels) - 1
        xr_channels = xr.DataArray([1] + [0] * l, dims=['channel'], coords=[channels])

        cav_array = cav_array * xr_channels

        cav_array = cav_array.transpose('subject', 'channel', 'y', 'x').astype(np.float32)

        cav_array.loc[dict(channel='walls')] = self.is_code(cav_array, PACAV.code_wall)
        cav_array.loc[dict(channel='crates')] = self.is_code(cav_array, PACAV.code_crate)
        cav_array.loc[dict(channel='coins')] = self.is_code(cav_array, PACAV.code_coin)
        cav_array.loc[dict(channel='agents')] = self.is_code(cav_array, PACAV.code_other_agent)
        cav_array.loc[dict(channel='bombs')] = self.is_code(cav_array, PACAV.code_bomb)

        cav_array.loc[dict(channel='btimes')] = self.bomb_time(cav_array)

        cav_array.loc[dict(channel='explosions')] = self.explosion_time(cav_array)

        channels = FeatureSelectionTransformationNCHW.channels[1:]
        cav_array = cav_array.loc[dict(channel=channels)]

        l_xds = xr.Dataset()
        l_xds['base'] = base_array
        l_xds['cav'] = cav_array

        return l_xds # base_array, cav_array
