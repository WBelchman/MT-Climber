
class EpisodeInfo():

    states = []
    actions = []
    rewards = [] 

    def __init__(self):
        pass

    def add(self, state, reward, action):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        
    def pop(self, index=0):
        s = self.states.pop(index=index)
        a = self.actions.pop(index=index)
        r = self.rewards.pop(index=index)

        return s, a, r

    def index(self, index):
        s = self.states[index]
        a = self.actions[index]
        r = self.rewards[index]

        return s, a, r
        