import gym
from RL_brain import DeepQNetwork

env = gym.make('MountainCar-v0')
env = env.unwrapped

print('这个环境中可用的 action 有 ', env.action_space)
print('这个环境中可用的 state 的 observation有 ', env.observation_space)
print('observation 的最高取值爲 ', env.observation_space.high)
print('observation 的最低取值爲 ', env.observation_space.low)

# 定義DQN的算法
RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9, replace_target_iter=300, memory_size=3000, e_greedy_increment=0.0002,)

# 記錄iterration
total_steps = 0

# 開始
for i_episode in range(1):

    # 获取回合 i_episode 的第一个 observation
    observation = env.reset()
    ep_r = 0
    while True:
        # 刷新環境
        env.render()

        # 利用RL選擇action
        action = RL.choose_action(observation)

        # 獲取下一個step
        observation_, reward, done, info = env.step(action)

        position, velocity = observation_

        # 車開得越高，reward越大
        reward = abs(position + 0.5)

        # 保存
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()

        ep_r += reward
        if done:
            get = '| Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
            print('Epi: ', i_episode, get, '| Ep_r: ', round(ep_r, 4), '| Epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1
        print(total_steps)

RL.plot_cost()
