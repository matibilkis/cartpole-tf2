import gym

env = gym.make("CartPole-v1")
for i in range(1):
    # print(help(env.render()))
    state = env.reset()

    done=False

    while done == False:
        env.render()

        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
