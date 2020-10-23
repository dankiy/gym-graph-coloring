import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from gym import spaces, Env

class NXColoringEnv(Env):
  def __init__(self, generator=nx.barabasi_albert_graph, **kwargs):
    '''
    generator — netwokrx graph generator,
    kwargs — generator named arguments
    '''
    self.G = generator(**kwargs)
    self.pos = nx.spring_layout(self.G, iterations=1000) #determine by n and m (?)
    self.edges = np.array(self.G.edges())
    self.n = len(self.G.nodes())
    self.m = len(self.edges)

    self.action_space = spaces.Box(low=0, high=self.n-1, shape=(self.n,2), dtype=np.uint32)
    self.used_colors = []
    self.current_state = np.full(self.n, self.n, dtype=np.uint32)
    self.done = False
    self.total_reward = 0

  def get_graph(self):
    return self.G.copy()

  def step(self, action):

    def is_action_available(action):
      node, color = action
      adjacent_nodes = np.unique(self.edges[np.sum(np.isin(self.edges, node), axis=1, dtype=bool)])
      return ~np.any(self.current_state[adjacent_nodes]==color)

    reward = 0
    
    if is_action_available(action):
      node, color = action
      self.current_state[node] = color
      if color not in self.used_colors:
        reward  = -1
        self.total_reward -= 1
        self.used_colors.append(color)

    if self.n not in np.unique(self.current_state):
      self.done = True
    
    info = {}

    return self.current_state, reward, self.done, info

  def reset(self):
    self.used_colors = []
    self.current_state = np.full(self.n, self.n, dtype=np.uint32)
    self.done = False
    self.total_reward = 0

  def render(self, mode='human', close=False):
    nx.draw(self.G, self.pos, node_color=self.current_state, cmap=plt.cm.tab20)