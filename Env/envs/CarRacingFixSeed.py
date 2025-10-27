from gymnasium.envs.box2d import CarRacing

class CarRacingFixSeed(CarRacing):
    def __init__(self,render_mode = None, index = 0):
        super().__init__(domain_randomize=True, continuous=True, render_mode=render_mode)
        self.reset_index = int(index)
    

    def reset(self, seed=None, options=None):
        return super().reset(seed=self.reset_index, options=options)



