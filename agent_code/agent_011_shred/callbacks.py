

import os,sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np

# import bm_data_collection_helper as dch
import central_arena_view as cav
from .model.model_base_mx import VGG20190317Model
import mxnet as mx

import pickle, time

def setup(self):
    self.logger.debug('Successfully entered setup code')
    self.logger.debug('mx.context.num_gpus: {}'.format(mx.context.num_gpus()))

    np.random.seed()
    # self.dch = dch.dch(self.logger, self.name, '0')

    self.m = VGG20190317Model(self.logger, auto_save=False)
    self.m.load()

    # prevent initial slow move during the game
    game_state_file_name = os.path.dirname(os.path.realpath(__file__)) + '/start-state.p'
    gs = pickle.load(open(game_state_file_name, "rb"))
    gs['self_score'] = (0, 0)

    time1 = time.time()
    Xdf, _ = self.m.get_transform().in_game_transform(gs)
    pQQs = self.m.predict(Xdf)
    time2 = time.time()
    self.logger.debug('Initial warmup prediction took: {} seconds'.format(time2 -time1))


def choice_max(self, pQQs, action_mask):
    pQQs = pQQs * action_mask
    # max_pQQ = np.max(pQQs)
    c = np.argmax(pQQs)
    next_action = self.m.action_options[c][0]
    return next_action, c, None

def choice_sample(self, pQQs, action_mask):
    pQQs = pQQs * action_mask
    weights = np.array(pQQs)
    weights = weights / np.sum(weights)
    c = np.random.choice(len(weights), p=weights)
    next_action = self.m.action_options[c][0]
    return next_action, c, weights

def adapt_pQQs0(self, pQQs):
    sc      = self.dch.step_count
    sinit   = 5
    sfactor = 2

    i = 0
    pQQs[i] -= max(0.0, (sinit- sc) * sfactor)
    pQQs[i]  = max(pQQs[i], 0.0)
    i = 5
    pQQs[i] -= max(0.0, (sinit- sc) * sfactor)
    pQQs[i]  = max(pQQs[i], 0.0)

    return pQQs

def adapt_pQQs(self, pQQs):
    self.logger.info('adapt_pQQs: before: {}'.format(pQQs))
    minQQ = np.min(pQQs)
    pQQs = pQQs - minQQ + 0.1
    self.logger.info('adapt_pQQs: after: {}'.format(pQQs))

    return pQQs

def choice_max_(self, pQQs, action_mask):
    pQQs = adapt_pQQs(self, pQQs)
    pQQs = pQQs * action_mask

    c = np.argmax(pQQs)
    next_action = self.m.action_options[c][0]
    return next_action, c, None

def choice_sample_(self, pQQs, action_mask):
    pQQs = adapt_pQQs(self, pQQs)
    pQQs = pQQs * action_mask

    weights = np.array(pQQs)
    weights = weights / np.sum(weights)
    c = np.random.choice(len(weights), p=weights)
    next_action = self.m.action_options[c][0]
    return next_action, c, weights


def prevent_most_stupid_moves_action_mask(self):
    self.logger.debug('Prevent most stupid moves: start')
    x, y, _, bombs_left, score = self.game_state['self']
    arena = self.game_state['arena']

    others = [(x, y) for (x, y, n, b, s) in self.game_state['others']]
    # bombs = self.game_state['bombs']
    # bomb_xys = [(x,y) for (x,y,t) in bombs]
    # bomb_map = np.ones(arena.shape) * 5
    # for xb,yb,t in bombs:
    #     for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
    #         if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
    #             bomb_map[i,j] = min(bomb_map[i,j], t)

    # A_ONE_HOT             = ['A_WAIT', 'A_UP', 'A_LEFT', 'A_DOWN', 'A_RIGHT', 'A_BOMB']
    action_mask = np.zeros(len(cav.DFIDs.A_ONE_HOT), dtype=np.float64)

    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                # (self.game_state['explosions'][d] <= 1) and
                # (bomb_map[d] > 0) and
                (not d in others)
                # and (not d in bomb_xys)
        ):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles:
        action_mask[2] = 1.0
        valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles:
        action_mask[4] = 1.0
        valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles:
        action_mask[1] = 1.0
        valid_actions.append('UP')
    if (x, y + 1) in valid_tiles:
        action_mask[3] = 1.0
        valid_actions.append('DOWN')
    if (x, y) in valid_tiles:
        action_mask[0] = 1.0
        valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0):
        action_mask[5] = 1.0
        valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')
    self.logger.debug('Prevent most stupid moves: end')
    return action_mask


def act(self):
    self.logger.info('Picking action according to rule set')
    # implementation of agent
    m = self.m

    self.logger.debug('Transforming game state: start')
    Xdf, av  = m.get_transform().in_game_transform(self.game_state)
    self.logger.debug('Transforming game state: finished')
    self.logger.debug('raw pQQs: start predict')
    pQQs = m.predict(Xdf)
    self.logger.debug('raw pQQs: {}'.format(pQQs))
    max_pQQ = np.max(pQQs)
    pQQs[1:] = np.maximum(0.1  , pQQs[1:])  # pQQs should be positive
    pQQs[:1] = np.maximum(0.01  , pQQs[:1]) # pQQs should be positive but wait should be discouraged even further
    pQQs = np.minimum(100.0, pQQs)
    self.logger.debug('pQQs cut with max and min: {}'.format(pQQs))

    # XXX if it is against the rule of the game to use such an action mask then just use only the ones line below instead of the following one
    action_mask = prevent_most_stupid_moves_action_mask(self)
    # action_mask = np.ones(len(cav.DFIDs.A_ONE_HOT), dtype=np.float64)
    self.logger.debug('action mask: {}'.format(action_mask))

    epsilon = 0.10
    r = np.random.random()
    if r < epsilon:
        self.logger.debug('selecting action based on sampling: r = {}, epsilon = {}'.format(r, epsilon))
        next_action, c, weights = choice_sample_(self, pQQs, action_mask)
    else:
        self.logger.debug('selecting action based on max: r = {}, epsilon = {}'.format(r,epsilon))
        next_action, c, weights = choice_max(self, pQQs, action_mask)
    # next_action, c, weights = choice_sample_(self, pQQs)

    self.logger.debug('weights: {}, choice: {}, next_action: {}, QQ: {}, max_QQ: {}'.format(weights, c, next_action, pQQs[c], max_pQQ))
    self.next_action = next_action

    # self.dch.callback01_register_game_state_and_action_av(av, self.next_action)

def reward_update(self):
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')
    # self.dch.callback02_process_events(self.game_state, self.events)

def end_of_episode(self):
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')
    # self.dch.callback03_end_game(self.game_state, self.events)
