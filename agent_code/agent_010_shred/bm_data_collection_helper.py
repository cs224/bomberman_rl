import os,sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import time, datetime, atexit
import numpy as np, pandas as pd
import central_arena_view as cav
from collections import namedtuple
from settings import s, e

# events = [
#     'MOVED_LEFT',
#     'MOVED_RIGHT',
#     'MOVED_UP',
#     'MOVED_DOWN',
#     'WAITED',
#     'INTERRUPTED',
#     'INVALID_ACTION',
#
#     'BOMB_DROPPED',
#     'BOMB_EXPLODED',
#
#     'CRATE_DESTROYED',
#     'COIN_FOUND',
#     'COIN_COLLECTED',
#
#     'KILLED_OPPONENT',
#     'KILLED_SELF',
#
#     'GOT_KILLED',
#     'OPPONENT_ELIMINATED',
#     'SURVIVED_ROUND',
# ]
# e = namedtuple('Events', events)(*range(len(events)))

class DataCollectionHelper():
    def __init__(self, logger, agent_id, game_run_id, file_name=None):
        self.logger = logger
        self.data_collection_id = game_run_id
        self.agent_id = agent_id
        self.numeric_agent_id = np.array(int(agent_id[-1]), dtype=np.byte)
        self.game_state_name = '{}-{}.h5'.format(self.data_collection_id, self.numeric_agent_id)

        self.cav_df = None
        self.av     = None
        self.step_count = 0
        self.game_count = 0
        self.file_name = file_name

        # file_name = './hdf5_training_data/' + self.game_state_name
        # self.store = pd.HDFStore(file_name)
        # atexit.register(lambda _ : self.store.close())

    def register_start_game(self):
        self.start_time = np.array(int(time.mktime(datetime.datetime.now().timetuple())), dtype=np.int32)

    def generate_ts(self):
        ts = np.array(int(time.mktime(datetime.datetime.now().timetuple())), dtype=np.int32)
        ts = ts - self.start_time
        return ts

    def callback01_register_game_state_and_action_av(self, av, str_action):
        self.logger.debug('DataCollectionHelper(): entering callback01_register_game_state_and_action_av')
        if self.cav_df is None:
            self.logger.debug('DataCollectionHelper(): registering start time')
            self.register_start_game()

        self.logger.debug('DataCollectionHelper(): creating PandasAugmentedCentralArenaView from game state')
        self.av = av
        self.av.set_data_collection_id(self.data_collection_id)
        self.av.set_time_stamp(self.generate_ts())
        self.av.set_game_count(self.game_count)
        self.av.set_action(str_action)
        self.av.set_agent_id(self.numeric_agent_id)
        self.logger.debug('DataCollectionHelper(): exiting callback01_register_game_state_and_action_av')


    def callback01_register_game_state_and_action(self, game_state, str_action):
        self.logger.debug('DataCollectionHelper(): entering callback01_register_game_state_and_action')
        if self.cav_df is None:
            self.logger.debug('DataCollectionHelper(): registering start time')
            self.register_start_game()

        self.logger.debug('DataCollectionHelper(): creating PandasAugmentedCentralArenaView from game state')
        self.av = cav.PandasAugmentedCentralArenaView(game_state)
        self.av.set_data_collection_id(self.data_collection_id)
        self.av.set_time_stamp(self.generate_ts())
        self.av.set_game_count(self.game_count)
        self.av.set_action(str_action)
        self.av.set_agent_id(self.numeric_agent_id)
        self.logger.debug('DataCollectionHelper(): exiting callback01_register_game_state_and_action')

    def extract_q_from_events(self, events):
        crate_destroyed = 0
        coin_found = 0
        coin_collected = 0
        opponent_eliminated = 0
        killed_opponent = 0
        got_killed = 0
        killed_self = 0
        penalty = 0

        for event in events:
            if event == e.CRATE_DESTROYED:
                crate_destroyed += 1
                continue

            if event == e.COIN_FOUND:
                coin_found += 1
                continue

            if event == e.COIN_COLLECTED:
                coin_collected += 1
                continue

            if event == e.OPPONENT_ELIMINATED:
                opponent_eliminated += 1
                continue

            if event == e.KILLED_OPPONENT:
                killed_opponent += 1
                continue

            if event == e.GOT_KILLED:
                got_killed += 1
                continue

            if event == e.KILLED_SELF:
                killed_self += 1
                continue

            if event == e.INTERRUPTED:
                penalty = penalty | 2 ** 1
                continue
            if event == e.INVALID_ACTION:
                penalty = penalty | 2 ** 2
                continue

        r = dict(crate_destroyed=crate_destroyed, coin_found=coin_found, coin_collected=coin_collected, opponent_eliminated=opponent_eliminated, killed_opponent=killed_opponent, got_killed=got_killed, killed_self=killed_self, penalty=penalty)
        return r

    def callback02_process_events(self, game_state, events):
        self.logger.debug('DataCollectionHelper(): entering callback02_process_events')
        if self.av is None:
            return

        q = self.extract_q_from_events(events)
        self.av.set_q_vars(**q)

        s = game_state['self_score'][0]
        self.av.set_q_score(s)

        self.av.set_game_step(self.step_count)
        self.step_count += 1
        ldf = self.av.to_df()

        if self.cav_df is None:
            self.cav_df = ldf
        else:
            self.cav_df = self.cav_df.append(ldf, ignore_index=True)
            self.logger.debug('DataCollectionHelper(): callback02_process_events : self.cav_df.shape: {}'.format(self.cav_df.shape))
        self.logger.debug('DataCollectionHelper(): exiting callback02_process_events')

    def callback03_end_game(self, game_state, events):
        self.logger.debug('DataCollectionHelper(): entering callback03_end_game')

        self.callback02_process_events(game_state, events)

        survived = e.SURVIVED_ROUND in events
        self.cav_df[cav.DFIDs.S] = survived

        score = game_state['self_score'][0]
        other_scores = [s[0] for s in game_state['others_scores']]
        max_other_scores = np.max(np.array(other_scores))

        # TODO the win logic seems to be wrong, e.g. in some cases callback03_end_game seems to be called not at the very end of a game, e.g.
        #   while your agent may be best at that moment later on it loses. The won state will be calculated in the post processing phase.
        # won = score >= max_other_scores
        won = False
        self.cav_df[cav.DFIDs.W] = won

        # self.store['df{}'.format(self.game_count)] = self.cav_df
        file_name = './hdf5_training_data/' + self.game_state_name
        if self.file_name is not None:
            file_name = self.file_name
        self.logger.debug('DataCollectionHelper(): creating: {} : {}'.format(file_name, '{}_{}'.format(self.agent_id, self.game_count)))
        self.cav_df.to_hdf(file_name, key='df_a{}_g{}'.format(self.numeric_agent_id, self.game_count), mode='a')
        # game_state_cav.to_hdf(game_state_name, key='{}_{}'.format(game_state_agent_name, game_state_count), mode='a')

        self.cav_df = None
        self.av     = None
        self.step_count = 0
        self.game_count += 1
        self.logger.debug('DataCollectionHelper(): exiting callback03_end_game')

def dch(logger, agent_id, game_run_id, file_name=None):
    return DataCollectionHelper(logger, agent_id, game_run_id, file_name=file_name)

