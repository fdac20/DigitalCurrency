
import numpy as np
#Class for a discrete (buy/hold/sell) spread trading environment.
class trade_enviroment():
    _actions = {
                    'hold': np.array([1, 0, 0]),
                    'buy': np.array([0, 1, 0]),
                    'sell': np.array([0, 0, 1])
                }
    _positions = {
                    'flat': np.array([1, 0, 0]),
                    'long': np.array([0, 1, 0]),
                    'short': np.array([0, 0, 1])
                }
    def __init__(self, data, parameter):
        self._data_generator = data
        self._first_render = True
        self._trading_fee = parameter[1]["trading_fee"]
        self._time_fee = parameter[1]["time_fee"]
        self._episode_length = parameter[1]["episode_length"]
        self.n_actions = 3
        self._prices_history = []
        self._history_length = 2
        self._tick_buy = 0
        self._tick_sell = 0
        self.tick_mid = 0
        self.tick_cci_14 = 0
        self.tick_rsi_14=0
        self.tick_dx_14 = 0
        self._price = 0
        self._round_digits = 4
        self._holding_position = []
        self._max_lost = -1000
        self._reward_factor = 10000
        self.reset()
        self.TP_render=False
        self.SL_render = False
        self.Buy_render=False
        self.Sell_render=False
        self.current_action="-"
        self.current_reward=0
        self.unr_pnl=0

    #Reset the trading environment
    def reset(self):
        self._iteration = 0
        self._data_generator.rewind()
        self._total_reward = 0
        self._total_pnl = 0
        self._current_pnl = 0
        self._position = self._positions['flat']

        self._closed_plot = False
        self._holding_position = []
        self._max_lost = -1000
        for i in range(self._history_length):
            self._prices_history.append(next(self._data_generator))
        self._tick_buy, self._tick_sell,self.tick_mid ,self.tick_rsi_14,self.tick_cci_14= \
            self._prices_history[0][:5]
        observation = self._get_observation()
        self.state_shape = observation.shape
        self._action = self._actions['hold']
        return observation
    #Take an action (buy/sell/hold) and calcultate the immediate reward.
    def step(self, action):
        self._action = action
        self._iteration += 1
        done = False
        info = {}
        if all(self._position != self._positions['flat']):
            reward = -self._time_fee
        self._current_pnl=0
        instant_pnl=0
        reward = -self._time_fee
        if all(action == self._actions['buy']):
            reward -= self._trading_fee
            if all(self._position == self._positions['flat']):
                self._position = self._positions['long']
                self._entry_price = self._price = self._tick_buy
                self.Buy_render = True
            elif all(self._position == self._positions['short']):
                self._exit_price = self._exit_price = self._tick_sell
                instant_pnl = self._entry_price - self._exit_price
                self._position = self._positions['flat']
                self._entry_price = 0
                # self.Buy_render = True
                if (instant_pnl > 0):
                    self.TP_render=True
                else:
                    self.SL_render=True

        elif all(action == self._actions['sell']):
            reward -= self._trading_fee
            if all(self._position == self._positions['flat']):
                self._position = self._positions['short']
                self._entry_price = self._price = self._tick_sell
                self.Sell_render = True
            elif all(self._position == self._positions['long']):
                self._exit_price = self._tick_buy
                instant_pnl = self._exit_price - self._entry_price
                self._position = self._positions['flat']
                self._entry_price = 0
                # self.Sell_render = True
                if (instant_pnl > 0):
                    self.TP_render = True
                else:
                    self.SL_render = True

        else:
            self.Buy_render = self.Sell_render = False
            self.TP_render = self.SL_render = False

        reward += instant_pnl
        self._total_pnl += instant_pnl
        self._total_reward += reward

        try:
            self._prices_history.append(next(self._data_generator))
            self._tick_sell, self._tick_buy, self.tick_mid, self.tick_rsi_14, self.tick_cci_14= \
            self._prices_history[-1][:5]
        except StopIteration:
            done = True
            info['status'] = 'No more data.'

        # Game over logic
        if self._iteration >= self._episode_length:
            done = True
            info['status'] = 'Time out.'
        if reward <= self._max_lost:
            done = True
            info['status'] = 'Bankrupted.'
        if self._closed_plot:
            info['status'] = 'Closed plot'

        observation = self._get_observation()

        return observation, reward, done, info
    #close position
    def _handle_close(self, evt):
        self._closed_plot = True

    #observe next state
    def _get_observation(self):
        if all(self._position==self._positions['flat']):
            self.unrl_pnl=0
        elif all(self._position==self._positions['long']):
            self.unrl_pnl = (self._prices_history[-1][2]-self._price)/self._prices_history[-1][2]
        elif all(self._position==self._positions['short']):
            self.unrl_pnl = (self._price - self._prices_history[-1][2])/self._prices_history[-1][2]

        return np.concatenate(
            [self._prices_history[-1][3:]] +
            [
                np.array([self.unrl_pnl]),
                np.array(self._position)
            ]
        )
