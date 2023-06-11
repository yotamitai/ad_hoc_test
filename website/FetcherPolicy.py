import numpy as np
import random
import copy


def never_query(obs, agent):
    return None


# Returns list of valid actions that brings fetcher closer to all tools
def get_valid_actions(obs, agent):
    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs

    valid_actions = np.array([True] * 4)  # NOOP is always valid
    for stn in range(len(s_pos)):
        if agent.probs[stn] == 0:
            continue
        tool_valid_actions = np.array([True] * 4)
        if f_pos[0] <= t_pos[stn][0]:
            tool_valid_actions[1] = False  # Left
        if f_pos[0] >= t_pos[stn][0]:
            tool_valid_actions[0] = False  # Right
        if f_pos[1] >= t_pos[stn][1]:
            tool_valid_actions[2] = False  # Down
        if f_pos[1] <= t_pos[stn][1]:
            tool_valid_actions[3] = False  # Up

        valid_actions = np.logical_and(valid_actions, tool_valid_actions)

    return valid_actions


class FetcherQueryPolicy:
    """
    Basic Fetcher Policy for querying, follows query_policy function argument (defaults to never query)
    Assumes all tools are in same location
    """

    def __init__(self, query_policy=never_query, prior=None, epsilon=0):
        self.query_policy = query_policy
        self._prior = prior
        self.probs = copy.deepcopy(self._prior)
        self.query = None
        self.prev_w_pos = None
        self._epsilon = epsilon

    def reset(self):
        self.probs = copy.deepcopy(self._prior)
        self.query = None
        self.prev_w_pos = None

    def make_inference(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.prev_w_pos is None:
            return
        if w_action == 5:
            for i, stn in enumerate(s_pos):
                if not np.array_equal(stn, self.prev_w_pos):
                    self.probs[i] *= self._epsilon
        elif w_action == 0:
            for i, stn in enumerate(s_pos):
                if stn[0] <= self.prev_w_pos[0]:
                    self.probs[i] *= self._epsilon
        elif w_action == 1:
            for i, stn in enumerate(s_pos):
                if stn[0] >= self.prev_w_pos[0]:
                    self.probs[i] *= self._epsilon
        elif w_action == 3:
            for i, stn in enumerate(s_pos):
                if stn[1] >= self.prev_w_pos[1]:
                    self.probs[i] *= self._epsilon
        elif w_action == 2:
            for i, stn in enumerate(s_pos):
                if stn[1] <= self.prev_w_pos[1]:
                    self.probs[i] *= self._epsilon

        self.probs /= np.sum(self.probs)

    def action_to_goal(self, pos, goal):
        actions = []
        if pos[0] < goal[0]:
            actions.append(0)
        elif pos[0] > goal[0]:
            actions.append(1)
        if pos[1] > goal[1]:
            actions.append(3)
        elif pos[1] < goal[1]:
            actions.append(2)
        if len(actions) == 0:
            return 4
        return np.random.choice(actions)

    def __call__(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.probs is None:
            self.probs = np.ones(len(s_pos))
            self.probs /= np.sum(self.probs)
        if answer is not None:
            if answer:
                for stn in range(len(s_pos)):
                    if stn not in self.query:
                        self.probs[stn] = 0
            else:
                for stn in self.query:
                    self.probs[stn] = 0
            self.probs /= np.sum(self.probs)
        else:
            self.make_inference(obs)

        self.prev_w_pos = np.array(w_pos)

        self.query = self.query_policy(obs, self)
        if self.query is not None:
            return 5, self.query

        if np.max(self.probs) < (1 - self._epsilon):
            # dealing with only one tool position currently
            if np.array_equal(f_pos, t_pos[0]):
                return 4, None
            else:
                return self.action_to_goal(f_pos, t_pos[0]), None
        else:
            if f_tool != np.argmax(self.probs):
                if np.array_equal(f_pos, t_pos[0]):
                    return 6, np.argmax(self.probs)
                else:
                    return self.action_to_goal(f_pos, t_pos[0]), None
            return self.action_to_goal(f_pos, s_pos[np.argmax(self.probs)]), None


class FetcherAltPolicy(FetcherQueryPolicy):
    """
    More Complicated Fetcher Policy, allows for multiple tool locations
    """

    def __call__(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.probs is None:
            self.probs = np.ones(len(s_pos))
            self.probs /= np.sum(self.probs)
        if answer is not None:
            if answer:
                for stn in range(len(s_pos)):
                    if stn not in self.query:
                        self.probs[stn] = 0
            else:
                for stn in self.query:
                    self.probs[stn] = 0
            self.probs /= np.sum(self.probs)
        else:
            self.make_inference(obs)

        self.prev_w_pos = np.array(w_pos)

        # One station already guaranteed. No querying needed.
        if np.max(self.probs) >= (1 - self._epsilon):
            target = np.argmax(self.probs)
            if f_tool != target:
                if np.array_equal(f_pos, t_pos[target]):
                    return 6, target
                else:
                    return self.action_to_goal(f_pos, t_pos[target]), None
            return self.action_to_goal(f_pos, s_pos[target]), None

        self.query = self.query_policy(obs, self)
        if self.query is not None:
            return 5, self.query

        # gets actions that are optimal even when not knowing the goal
        valid_actions = get_valid_actions(obs, self)

        if np.any(valid_actions):
            print(valid_actions)
            p = valid_actions / np.sum(valid_actions)
            action_idx = np.random.choice(np.arange(4), p=p)
            return action_idx, None
        else:
            return 4, None
