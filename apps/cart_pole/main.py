import gymnasium as gym
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter 


img_dir = './images/'
def display_frames_as_gif(frames, gif_filename='./ppo_cartpolev0_result.gif'):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
                                    
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save(img_dir+gif_filename, writer='imagemagick', fps=30)

frames = []

#env = gym.make('CartPole-v1', render_mode="human")
env = gym.make('CartPole-v1', render_mode="rgb_array")
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        frames.append(env.render())

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()

display_frames_as_gif(frames, "test.gif")


