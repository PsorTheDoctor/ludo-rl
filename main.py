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


def start_teaching_ai_agent(agent, episodes, n_players, epsilon, epsilon_decay_rate, lr, gamma,
                            verbose=True, display=True):
    # Housekeeping variables
    ai_player_winning_avg = []
    epsilon_list = []
    idx = []
    ai_player_won = 0

    # Store data
    win_rate_list = []
    max_expected_return_list = []

    if n_players == 4:
        g = ludopy.Game(ghost_players=[])
    elif n_players == 3:
        g = ludopy.Game(ghost_players=[1])
    elif n_players == 2:
        g = ludopy.Game(ghost_players=[1, 3])
    else:
        print('Number of players must be 2, 3 or 4!')

    if agent == 'q_learning':
        ai_player = QLearningAgent(2, learning_rate=lr, gamma=gamma)
    elif agent == 'double_q_learning':
        ai_player = DoubleQLearningAgent(2, learning_rate=lr, gamma=gamma)
    # elif agent == 'distrib_q_learning':
    #     ai_player = DistribQLearningAgent(2, learning_rate=lr, gamma=gamma)
    elif agent == 'sarsa':
        ai_player = SarsaAgent(2, learning_rate=lr, gamma=gamma)
    elif agent == 'expected_sarsa':
        ai_player = ExpectedSarsaAgent(2, learning_rate=lr, gamma=gamma)
    elif agent == 'td0':
        ai_player = TD0Agent(2, learning_rate=lr, gamma=gamma)
    else:
        print('Agent must be specified!')

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
            
            if display and episode > 1:
                board = g.render_environment()
                cv2.imshow("Ludo Board", cv2.resize(board, (0, 0), fx=0.5, fy=0.5))
                cv2.waitKey(20)
                
            if ai_player.ai_player_idx == player_i and piece_to_move != -1:
                ai_player.reward(g.players, [piece_to_move])

        if display and episode == 200:
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

        if verbose and episode % 1 == 0:
            print('Players: {} episode: {} win rate: {}%'.format(n_players,
                                                                 episode + 1,
                                                                 np.round(win_rate_percentage, 1)))
    
        max_expected_return_list.append(ai_player.rewards.max_expected_reward)
        ai_player.rewards.max_expected_reward = 0

    # Moving averages
    window_size = int(0.1 * episodes)
    cumsum_vec = np.cumsum(np.insert(win_rate_list, 0, 0)) 
    win_rate_ma = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

    cumsum_vec = np.cumsum(np.insert(max_expected_return_list, 0, 0)) 
    max_expected_return_list_ma = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

    moving_average_list = [0] * window_size
    win_rate_ma = moving_average_list + win_rate_ma.tolist()
    max_expected_return_list_ma = moving_average_list + max_expected_return_list_ma.tolist()

    # display_q_table(ai_player.rewards.q_table)
    # if agent == 'double_q_learning':
    #     display_q_table(ai_player.rewards.q2_table)

    return win_rate_list, win_rate_ma, epsilon_list, max_expected_return_list, max_expected_return_list_ma


def teach_agent(agent):
    win_rates = []
    # win_rate_mas = []
    max_expected_returns = []
    epsilons = []

    for n_players in range(2, 5):
        win_rate, win_rate_ma, epsilons, max_expected_return, max_expected_return_ma = start_teaching_ai_agent(
            agent, episodes, n_players, epsilon, epsilon_decay_rate, lr, gamma, display=False
        )
        win_rates.append(win_rate)
        # win_rate_mas.append(win_rate_ma)
        max_expected_returns.append(max_expected_return_ma)
        epsilons.append(epsilon)

    win_rates = np.asarray(win_rates)
    # win_rate_mas = np.asarray(win_rate_mas)
    max_expected_returns = np.asarray(max_expected_returns)
    return win_rates, max_expected_returns, epsilons


# FINAL AGENT PLAYING AGAINST 1, 2 and 3 RANDOM PLAYERS

lr = 0.2
gamma = 0.5
epsilon = 0.9
epsilon_decay_rate = 0.05
episodes = 1000
"""
win_rates = []
for lr in [0.1, 0.2, 0.3]:
    win_rate, win_rate_ma, epsilons, max_expected_return, max_expected_return_ma = start_teaching_ai_agent(
        'q_learning', episodes, 4, epsilon, epsilon_decay_rate, lr, gamma, display=False
    )
    win_rates.append(win_rate)
    # print('lr={}: {}'.format(lr, np.mean(win_rate)))
"""
# Start teaching agents
# q_learning_win_rates, q_learning_max_exp_returns, epsilons = teach_agent('q_learning')
# dq_learning_win_rates, dq_learning_max_exp_returns, _ = teach_agent('double_q_learning')
exp_sarsa_win_rates, sarsa_max_exp_returns, _ = teach_agent('expected_sarsa')
# exp_sarsa_win_rates, exp_sarsa_max_exp_returns, _ = teach_agent('expected_sarsa')

#exp_sarsa_means = [np.mean(exp_sarsa_win_rates[0]), np.mean(exp_sarsa_win_rates[1]), np.mean(exp_sarsa_win_rates[2])]
#exp_sarsa_stds = [np.std(exp_sarsa_win_rates[0]), np.std(exp_sarsa_win_rates[1]), np.std(exp_sarsa_win_rates[2])]

fig, axs = plt.subplots(1)
axs.set_title("Win Rate against different number of opponents")
axs.set_xlabel('Episodes')
axs.set_ylabel('Win Rate %')
axs.plot(sarsa_win_rates[0], label='1 Opponent', color='tab:red')
axs.plot(sarsa_win_rates[1], label='2 Opponents')
axs.plot(sarsa_win_rates[2], label='3 Opponents')
axs.legend()
plt.show()

"""
for n in range(3):
    fig, ax = plt.subplots(1)
    if n == 0:
        ax.set_title('Win Rate against 1 opponent')
    else:
        ax.set_title('Win Rate against {} opponents'.format(n + 1))
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Win Rate %')
    ax.plot(q_learning_win_rates[n], label='Q-learning')
    ax.plot(dq_learning_win_rates[n], label='Double Q-learning')
    ax.plot(sarsa_win_rates[n], label='SARSA')
    ax.plot(exp_sarsa_win_rates[n], label='Expected SARSA')
    ax.legend()

    # fig, ax = plt.subplots(1)
    # if n == 0:
    #     ax.set_title('Win Rate moving average against 1 opponent')
    # else:
    #     ax.set_title('Win Rate moving average against {} opponents'.format(n + 1))
    # ax.set_xlabel('Episodes')
    # ax[1].set_ylabel('Win Rate %')
    # ax.plot(q_learning_win_rate_mas[n], label='Q-learning')
    # ax.plot(dq_learning_win_rate_mas[n], label='Double Q-learning')
    # ax.plot(sarsa_win_rate_mas[n], label='SARSA')
    # ax.legend()

    fig, ax = plt.subplots(1)
    if n == 0:
        ax.set_title('Max expected return against 1 opponent')
    else:
        ax.set_title('Max expected return against {} opponents'.format(n + 1))
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Max expected return')
    ax.plot(q_learning_max_exp_returns[n], label='Q-learning')
    ax.plot(dq_learning_max_exp_returns[n], label='Double Q-learning')
    ax.plot(sarsa_max_exp_returns[n], label='SARSA')
    ax.plot(exp_sarsa_max_exp_returns[n], label='Expected SARSA')
    ax.legend()

plt.show()

fig, axs = plt.subplots(1)
axs.set_title("Epsilon Decay")
axs.set_xlabel('Episodes')
axs.set_ylabel('Epsilon')
axs.plot(epsilons[:-1], color='tab:red')
axs.legend(['Epsilon Decay 0.05'])

q_learning_means = [np.mean(q_learning_win_rates[0]), np.mean(q_learning_win_rates[1]), np.mean(q_learning_win_rates[2])]
dq_learning_means = [np.mean(dq_learning_win_rates[0]), np.mean(dq_learning_win_rates[1]), np.mean(dq_learning_win_rates[2])]
sarsa_means = [np.mean(sarsa_win_rates[0]), np.mean(sarsa_win_rates[1]), np.mean(sarsa_win_rates[2])]
exp_sarsa_means = [np.mean(exp_sarsa_win_rates[0]), np.mean(exp_sarsa_win_rates[1]), np.mean(exp_sarsa_win_rates[2])]

q_learning_stds = [np.std(q_learning_win_rates[0]), np.std(q_learning_win_rates[1]), np.std(q_learning_win_rates[2])]
dq_learning_stds = [np.std(dq_learning_win_rates[0]), np.std(dq_learning_win_rates[1]), np.std(dq_learning_win_rates[2])]
sarsa_stds = [np.std(sarsa_win_rates[0]), np.std(sarsa_win_rates[1]), np.std(sarsa_win_rates[2])]
exp_sarsa_stds = [np.std(exp_sarsa_win_rates[0]), np.std(exp_sarsa_win_rates[1]), np.std(exp_sarsa_win_rates[2])]

for n in range(3):
    print('Win rate against {} opponents'.format(n + 1))
    print('Q-learning: {} mean: {} std: {}'.format(
        q_learning_win_rates[n][-1], q_learning_means[n], q_learning_stds[n]
    ))
    print('Double Q-learning: {} mean: {} std: {}'.format(
        dq_learning_win_rates[n][-1], dq_learning_means[n], dq_learning_stds[n]
    ))
    print('SARSA: {} mean: {} std: {}'.format(
        sarsa_win_rates[n][-1], sarsa_means[n], sarsa_stds[n]
    ))
    print('Expected SARSA: {} mean: {} std: {}'.format(
        exp_sarsa_win_rates[n][-1], exp_sarsa_means[n], exp_sarsa_stds[n]
    ))

fig, axs = plt.subplots(1)
axs.set_title('Win Rate comparision')
axs.set_xlabel('Opponents')
axs.set_ylabel('Win Rate %')
axs.set_xticks(np.arange(3), labels=['1', '2', '3'])
axs.bar(np.arange(3) - 0.3, q_learning_means, 0.2, label='Q-learning', color='#28784b')
axs.bar(np.arange(3) - 0.1, dq_learning_means, 0.2, label='Double Q-learning', color='seagreen')
axs.bar(np.arange(3) + 0.1, sarsa_means, 0.2, label='SARSA', color='mediumseagreen')
axs.bar(np.arange(3) + 0.3, exp_sarsa_means, 0.2, label='Expected SARSA', color='darkseagreen')
axs.errorbar(np.arange(3) - 0.3, q_learning_means, yerr=q_learning_stds,
             fmt=' ', linewidth=2, capsize=10, color='tab:red')
axs.errorbar(np.arange(3) - 0.1, dq_learning_means, yerr=dq_learning_stds,
             fmt=' ', linewidth=2, capsize=10, color='tab:red')
axs.errorbar(np.arange(3) + 0.1, sarsa_means, yerr=sarsa_stds,
             fmt=' ', linewidth=2, capsize=10, color='tab:red')
axs.errorbar(np.arange(3) + 0.3, exp_sarsa_means, yerr=exp_sarsa_stds,
             fmt=' ', linewidth=2, capsize=10, color='tab:red')
# axs.plot(np.ones(3) * 25, linestyle='--', color='crimson')
# axs.plot(np.ones(3) * 33, linestyle='--', color='crimson')
# axs.plot(np.ones(3) * 5, linestyle='--', color='crimson')
axs.legend()

plt.show()
"""
