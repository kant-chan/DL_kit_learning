from maze_env import Maze
from RL_brain import QLearningTable, SarsaTable


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))
            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

def update_sarsa():
    for episode in range(100):
        observation = env.reset()
        action = RL.choose_action(str(observation))
        
        # lambda 版本记得每回合清零
        # RL.eligibility_trace *= 0
        while True:
            env.render()
            observation_, reward, done = env.step(action)
            action_ = RL.choose_action(str(observation_))
            RL.learn(str(observation), action, reward, str(observation_), action_)
            observation, action = observation_, action_
            action = action_

            if done:
                break
    
    print('game over')
    print(RL.q_table)
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    # RL = QLearningTable(actions=list(range(env.n_actions)))
    # env.after(100, update)
    RL = SarsaTable(actions=list(range(env.n_actions)))
    env.after(100, update_sarsa)
    env.mainloop()
