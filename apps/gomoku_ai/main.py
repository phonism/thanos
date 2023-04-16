import gym
import random
import gym_gomoku
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import animation

img_dir = './images/'
def display_frames_as_gif(frames, gif_filename='./none.gif'):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
                                    
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save(img_dir+gif_filename, writer='imagemagick', fps=30)

class FFN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FFN, self).__init__()
        self.hidden_dim = 128

        self.layers = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, out_dim),
                #nn.Dropout(0.5),
                nn.Softmax(dim=-1)
        )

    def forward(self, observation):
        return self.layers(observation)

class PolicyGradient(object):
    def __init__(self, env):
        self.n_updates = 10
        self.lr = 0.01
        self.gamma = 0.99

        self.env = env
        self.obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1] # 观测/状态空间
        self.act_dim = env.action_space.n # 动作空间
        self.model = FFN(self.obs_dim, self.act_dim)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def train(self):
        print('*'*8, 'start learning...')
        self.model.train()
        max_rewards = 0
        for eps in range(10000):
            self.optim.zero_grad()
            batch_observation, batch_action, batch_reward, batch_log_probs = self.rollout(self.n_updates)
            batch_reward_togo = self.compute_reward_togo(batch_reward)

            batch_reward_togo = torch.Tensor(batch_reward_togo)
            batch_log_probs = torch.concat(batch_log_probs, dim=0)

            reward_mean = torch.mean(batch_reward_togo)
            reward_std = torch.std(batch_reward_togo)

            batch_reward_togo = (batch_reward_togo - reward_mean) / (reward_std + 1e-10)

            loss = (-batch_log_probs * batch_reward_togo).sum() / self.n_updates

            loss.backward()
            self.optim.step()
            print(sum(map(sum, batch_reward)) / self.n_updates, loss.detach().item())
            for param in self.model.parameters():
                #print(param.requires_grad)
                continue

        torch.save(self.model, "models/policy_gradient")

        self.rollout(1, render=True)


    def get_action(self, obs):
        obs = obs.view(-1)
        probs = self.model(obs)
        # 注意：在test阶段，action是确定的，即就是上面的"probs"，但train阶段actor应该是去尽量『探索』各种actions，为了模拟这个探索的过程，
        # 我们借助概率分布来模拟
        dist = torch.distributions.Categorical(probs) # 以概率probs形成一个类别分布
        action = dist.sample([1]) # 指定生成样本的维度 # [1]
        log_prob = dist.log_prob(action) # [1]
        return action.item(), log_prob

    def rollout(self, times, render=False):
        batch_observation = []
        batch_log_probs = []
        batch_action = []
        batch_reward = []
        
        frames = []
        for batch in range(times):
            reward_list = []
            observation = self.env.reset()
            for t in range(100000):
                if render:
                    frames.append(self.env.render())
                batch_observation.append(observation)
                action, log_prob = self.get_action(torch.Tensor(observation))
                batch_log_probs.append(log_prob)
                if torch.Tensor(observation).view(-1)[action] != 0:
                    reward = -1000
                    done = True
                else:
                    observation, reward, done, info = self.env.step(action)
                if reward == 1:
                    #self.env.render()
                    reward = 1000
                if reward == 0:
                    reward = 1
                if reward == -1:
                    reward = -20
                batch_action.append(action)
                reward_list.append(reward)
                if done:
                    #if random.randint(1, 1000) == 77:
                        #self.env.render()
                    break
            batch_reward.append(reward_list)
        if render:
            display_frames_as_gif(frames, "test.gif")

        return batch_observation, batch_action, batch_reward, batch_log_probs

    def compute_reward_togo(self, batch_reward):
        batch_reward_togo = []
        for reward_list in batch_reward[::-1]:
            discounted_reward = 0
            for reward in reward_list[::-1]:
                discounted_reward = reward + discounted_reward * self.gamma
                batch_reward_togo.append(discounted_reward)
        batch_reward_togo.reverse()
        return batch_reward_togo


if __name__ == "__main__":
    #env = gym.make("Gomoku9x9-v0")
    env = gym.make("Gomoku6x6-v0")
    pg = PolicyGradient(env)
    pg.train()

