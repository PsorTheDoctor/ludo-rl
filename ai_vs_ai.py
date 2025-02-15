import ludopy
import matplotlib.pyplot as plt
import numpy as np
import cv2
from player import *
from utils import *


def movingaverage(interval, window_size):
  window = np.ones(int(window_size)) / float(window_size)
  return np.convolve(interval, window, 'same')


def epsilon_decay(epsilon, decay_rate, episode):
  return epsilon * np.exp(-decay_rate * episode)


def ai_vs_ai(episodes, epsilon, epsilon_decay_rate, lr, gamma, verbose=True, display=True):
    # Housekeeping variables
    ai_player_winning_avg = {'sarsa': [], 'ql': [], 'dql': []}
    epsilon_list = []
    idx = []
    ai_player_won = {'sarsa': 0, 'ql': 0, 'dql': 0}

    # Store data
    win_rate_list = {'sarsa': [], 'ql': [], 'dql': []}
    win_rate_ma = {'sarsa': [], 'ql': [], 'dql': []}
    max_expected_return_list = []

    g = ludopy.Game(ghost_players=[])
    ai_player1 = QLearningAgent(1, learning_rate=lr, gamma=gamma)
    ai_player2 = DoubleQLearningAgent(3, learning_rate=lr, gamma=gamma)
    # ai_player3 = SarsaAgent(3, learning_rate=lr, gamma=gamma)

    for episode in range(0, episodes):
      there_is_a_winner = False
      g.reset()
      while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()

        if len(move_pieces):
          if ai_player1.ai_player_idx == player_i:
            piece_to_move = ai_player1.update(g.players, move_pieces, dice)
            if not piece_to_move in move_pieces:
              g.render_environment()
          if ai_player2.ai_player_idx == player_i:
            piece_to_move = ai_player2.update(g.players, move_pieces, dice)
            if not piece_to_move in move_pieces:
              g.render_environment()
          # if ai_player3.ai_player_idx == player_i:
          #   piece_to_move = ai_player3.update(g.players, move_pieces, dice)
            if not piece_to_move in move_pieces:
              g.render_environment()
          else:
            piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
        else:
          piece_to_move = -1
        _, _, _, _, playerIsAWinner, there_is_a_winner = g.answer_observation(piece_to_move)

        if display and episode > 1:
          board = g.render_environment()
          cv2.imshow("Ludo Board", cv2.resize(board, (0, 0), fx=0.5, fy=0.5))
          cv2.waitKey(20)

        if ai_player1.ai_player_idx == player_i and piece_to_move != -1:
          ai_player1.reward(g.players, [piece_to_move])
        elif ai_player2.ai_player_idx == player_i and piece_to_move != -1:
          ai_player2.reward(g.players, [piece_to_move])
        # elif ai_player3.ai_player_idx == player_i and piece_to_move != -1:
        #   ai_player3.reward(g.players, [piece_to_move])

      if display and episode == 200:
        g.save_hist_video("game.mp4")

      new_epsilon_after_decay = epsilon_decay(epsilon=epsilon, decay_rate=epsilon_decay_rate, episode=episode)
      epsilon_list.append(new_epsilon_after_decay)
      ai_player1.rewards.update_epsilon(new_epsilon_after_decay)
      ai_player2.rewards.update_epsilon(new_epsilon_after_decay)
      # ai_player3.rewards.update_epsilon(new_epsilon_after_decay)

      if g.first_winner_was == ai_player1.ai_player_idx:
        # ai_player_winning_avg['sarsa'].append(1)
        ai_player_won['ql'] = ai_player_won['ql'] + 1
        ai_player_winning_avg['ql'].append(1)
        ai_player_winning_avg['dql'].append(0)

      elif g.first_winner_was == ai_player2.ai_player_idx:
        ai_player_winning_avg['dql'].append(1)
        ai_player_won['dql'] = ai_player_won['dql'] + 1
        # ai_player_winning_avg['sarsa'].append(0)
        ai_player_winning_avg['ql'].append(0)

      # elif g.first_winner_was == ai_player3.ai_player_idx:
      #   ai_player_winning_avg['dql'].append(1)
      #   ai_player_won['dql'] = ai_player_won['dql'] + 1
      #   ai_player_winning_avg['sarsa'].append(0)
      #   ai_player_winning_avg['ql'].append(0)

      idx.append(episode)

      # Moving averages
      window_size = 50

      for player in ['ql', 'dql']:
        win_rate = ai_player_won[player] / len(ai_player_winning_avg[player])
        win_rate_percentage = win_rate * 100
        win_rate_list[player].append(win_rate_percentage)

        cumsum_vec = np.cumsum(np.insert(win_rate_list[player], 0, 0))
        win_rate_ma[player] = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
        moving_average_list = [0] * window_size
        win_rate_ma[player] = moving_average_list + win_rate_ma[player].tolist()

      if verbose and episode % 1 == 0:
        print('episode: {} ql: {}% dql: {}%'.format(episode + 1,
                                                               # np.round(win_rate_list['sarsa'][-1], 1),
                                                               np.round(win_rate_list['ql'][-1], 1),
                                                               np.round(win_rate_list['dql'][-1], 1)))
      # max_expected_return_list.append(ai_player.rewards.max_expected_reward)
      # ai_player.rewards.max_expected_reward = 0

    return win_rate_list, win_rate_ma


lr = 0.19
gamma = 0.25
epsilon = 0.35
epsilon_decay_rate = 0.05
episodes = 500

win_rate, win_rate_ma = ai_vs_ai(episodes, epsilon, epsilon_decay_rate, lr, gamma, display=False)

fig, axs = plt.subplots(1)
axs.set_title('Win Rate of the agents against themselves')
axs.set_xlabel('Episodes')
axs.set_ylabel('Win Rate %')
axs.plot(win_rate['ql'], label='Q-learning')
axs.plot(win_rate['dql'], label='Double Q-learning')
# axs.plot(win_rate['sarsa'], label='SARSA')
axs.legend()

fig, axs = plt.subplots(1)
axs.set_title('Win Rate moving average of the agents against themselves')
axs.set_xlabel('Episodes')
axs.set_ylabel('Win Rate %')
axs.plot(win_rate_ma['ql'], label='Q-learning')
axs.plot(win_rate_ma['dql'], label='Double Q-learning')
# axs.plot(win_rate_ma['sarsa'], label='SARSA')
axs.legend()

plt.show()
