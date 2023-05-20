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

def start_teaching_ai_agent(agent, episodes, no_of_players, epsilon, epsilon_decay_rate, lr, gamma, display=True):
    
    # Housekeeping variables
    ai_player_winning_avg = []
    epsilon_list = []
    idx = []
    ai_player_won = 0

    # Store data
    win_rate_list = []
    max_expected_return_list = []

    if no_of_players == 4:
        g = ludopy.Game(ghost_players=[])
    elif no_of_players == 3:
        g = ludopy.Game(ghost_players=[1])
    elif no_of_players == 2:
        g = ludopy.Game(ghost_players=[1, 3])
    else:
        print('Number of players must be 2, 3 or 4!')

    if agent == 'td0':
        ai_player = TD0Agent(2, learning_rate=lr, gamma=gamma)
    if agent == 'q_learning':
        ai_player = QLearningAgent(2, learning_rate=lr, gamma=gamma)
    elif agent == 'sarsa':
        ai_player = SarsaAgent(2, learning_rate=lr, gamma=gamma)
    else:
        print('Agent must be td0, q_learning or sarsa!')

    for episode in range(0, episodes):

        there_is_a_winner = False
        g.reset()
        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,there_is_a_winner), player_i = g.get_observation()

            if len(move_pieces):
                if ai_player.ai_player_idx == player_i:
                    piece_to_move = ai_player.update(g.players, move_pieces, dice)
                    if not piece_to_move in move_pieces:
                        g.render_environment()
                else:
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1
            _, _, _, _, playerIsAWinner, there_is_a_winner = g.answer_observation(piece_to_move)
            
            if episode and display > 1:
                board = g.render_environment()
                cv2.imshow("Ludo Board", cv2.resize(board, (0, 0), fx=0.5, fy=0.5))
                cv2.waitKey(20)
                
            if ai_player.ai_player_idx == player_i and piece_to_move != -1:
                ai_player.reward(g.players, [piece_to_move])

        if episode and display == 200:
            g.save_hist_video("game.mp4")
            
        new_epsilon_after_decay = epsilon_decay(epsilon=epsilon, decay_rate=epsilon_decay_rate, episode=episode)
        epsilon_list.append(new_epsilon_after_decay)
        ai_player.rewards.update_epsilon(new_epsilon_after_decay)

        if g.first_winner_was == ai_player.ai_player_idx:
            ai_player_winning_avg.append(1)
            ai_player_won = ai_player_won + 1
        else:
            ai_player_winning_avg.append(0)

        idx.append(episode)

        # Print some results
        win_rate = ai_player_won / len(ai_player_winning_avg)
        win_rate_percentage = win_rate * 100
        win_rate_list.append(win_rate_percentage)

        if episode % 1 == 0:
            print('Players: {} episode: {} win rate: {}%'.format(no_of_players,
                                                                 episode + 1,
                                                                 np.round(win_rate_percentage, 1)))
    
        max_expected_return_list.append(ai_player.rewards.max_expected_reward)
        ai_player.rewards.max_expected_reward = 0

    # Moving averages
    window_size = 20
    cumsum_vec = np.cumsum(np.insert(win_rate_list, 0, 0)) 
    win_rate_ma = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

    cumsum_vec = np.cumsum(np.insert(max_expected_return_list, 0, 0)) 
    max_expected_return_list_ma = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

    moving_average_list = [0] * window_size
    win_rate_ma = moving_average_list + win_rate_ma.tolist()
    max_expected_return_list_ma = moving_average_list + max_expected_return_list_ma.tolist()

    # display_q_table(ai_player.rewards.q_table)

    return win_rate_list, win_rate_ma, epsilon_list, max_expected_return_list, max_expected_return_list_ma


# FINAL AGENT PLAYING AGAINST 1, 2 and 3 RANDOM PLAYERS

learning_rate = 0.2
gamma = 0.5
epsilon = 0.9
epsilon_decay_rate = 0.05
episodes = 30

q_learning_win_rates = []
q_learning_win_rate_mas = []
sarsa_win_rates = []
sarsa_win_rate_mas = []

# Start teaching Q-learning agent
for n_players in range(2, 5):
    win_rate, win_rate_ma, epsilons, max_expected_return, max_expected_return_ma = start_teaching_ai_agent(
        'q_learning', episodes, n_players, epsilon, epsilon_decay_rate, learning_rate, gamma, display=False
    )
    q_learning_win_rates.append(win_rate)
    q_learning_win_rate_mas.append(win_rate_ma)

q_learning_win_rates = np.asarray(q_learning_win_rates)
q_learning_win_rate_mas = np.asarray(q_learning_win_rate_mas)

# Start teaching SARSA agent
for n_players in range(2, 5):
    win_rate, win_rate_ma, epsilons, max_expected_return, max_expected_return_ma = start_teaching_ai_agent(
        'sarsa', episodes, n_players, epsilon, epsilon_decay_rate, learning_rate, gamma, display=False
    )
    sarsa_win_rates.append(win_rate)
    sarsa_win_rate_mas.append(win_rate_ma)

sarsa_win_rates = np.asarray(sarsa_win_rates)
sarsa_win_rate_mas = np.asarray(sarsa_win_rate_mas)

fig, axs = plt.subplots(1)
axs.set_title('Q-learning Win Rate')
axs.set_xlabel('Episodes')
axs.set_ylabel('Win Rate %')
axs.plot(q_learning_win_rates[0], label='1 Opponent')
axs.plot(q_learning_win_rates[1], label='2 Opponents')
axs.plot(q_learning_win_rates[2], label='3 Opponents')
axs.legend()

fig, axs = plt.subplots(1)
axs.set_title('Q-learning Win Rate moving average')
axs.set_xlabel('Episodes')
axs.set_ylabel('Win Rate %')
axs.plot(q_learning_win_rate_mas[0], label='1 Opponent')
axs.plot(q_learning_win_rate_mas[1], label='2 Opponents')
axs.plot(q_learning_win_rate_mas[2], label='3 Opponents')
axs.legend()

fig, axs = plt.subplots(1)
axs.set_title('SARSA Win Rate')
axs.set_xlabel('Episodes')
axs.set_ylabel('Win Rate %')
axs.plot(sarsa_win_rates[0], label='1 Opponent')
axs.plot(sarsa_win_rates[1], label='2 Opponents')
axs.plot(sarsa_win_rates[2], label='3 Opponents')
axs.legend()

fig, axs = plt.subplots(1)
axs.set_title('SARSA Win Rate moving average')
axs.set_xlabel('Episodes')
axs.set_ylabel('Win Rate %')
axs.plot(sarsa_win_rate_mas[0], label='1 Opponent')
axs.plot(sarsa_win_rate_mas[1], label='2 Opponents')
axs.plot(sarsa_win_rate_mas[2], label='3 Opponents')
axs.legend()

fig, axs = plt.subplots(1)
axs.set_title("Epsilon Decay")
axs.set_xlabel('Episodes')
axs.set_ylabel('Epsilon')
axs.plot(epsilons, color='tab:red')
axs.legend(['Epsilon Decay 0.05'])

q_learning_means = [
    np.mean(q_learning_win_rates[0]), np.mean(q_learning_win_rates[1]), np.mean(q_learning_win_rates[2])
]
sarsa_means = [
    np.mean(sarsa_win_rates[0]), np.mean(sarsa_win_rates[1]), np.mean(sarsa_win_rates[2])
]
q_learning_stds = [
    np.std(q_learning_win_rates[0]), np.std(q_learning_win_rates[1]), np.std(q_learning_win_rates[2])
]
sarsa_stds = [
    np.std(sarsa_win_rates[0]), np.std(sarsa_win_rates[1]), np.std(sarsa_win_rates[2])
]

fig, axs = plt.subplots(1)
axs.set_title('')
axs.bar(np.arange(3) - 0.2, q_learning_means, 0.4, label='Q-learning', color='seagreen')
axs.bar(np.arange(3) + 0.2, sarsa_means, 0.4, label='SARSA', color='mediumseagreen')
plt.errorbar(np.arange(3) - 0.2, q_learning_means, yerr=q_learning_stds,
             fmt=' ', linewidth=2, capsize=20, color='tab:red')
plt.errorbar(np.arange(3) + 0.2, sarsa_means, yerr=sarsa_stds,
             fmt=' ', linewidth=2, capsize=20, color='tab:red')
axs.legend()

plt.show()
