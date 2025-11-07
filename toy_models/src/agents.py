import gymnasium as gym
from collections import defaultdict

class QlearnAgent:
    def __init__(self, env, lr, eps_init, eps_decay, eps_final, discount_factor):
        self.env = ev
        
        # learning hyperparams
        self.lr = lr # rate at which to scale q-val
        self.discount_factor = discount_factor
        self.eps = eps_init # parameterizes greediness
        self.eps_decay = eps_decay # linear decay factor
        self.eps_final = eps_final # low limit eps

        self.training_error = []
        
        # q (action-value) function
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, obs):
        # if less than eps, explore, else exploit
        if np.random.random() < self.eps:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):

        future_q_val = np.max(self.q_values[next_obs]) if not terminated else 0

        # bellman eq'n
        target = reward + self.discount_factor * future_q_val

        # temporal diff
        temp_diff = target - self.q_values[obs][action]

        # update q
        self.q_values[obs][action] = (self.q_values[obs][action] + self.lr * temp_diff)

        # track training error
        self.training_error.append(temp_diff)

    def decay_epsilon(self):
        self.eps = max(self.final_eps, self.eps - self.eps_decay)