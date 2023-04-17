import gym
import random
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import animation

#random.seed(10)
#torch.manual_seed(0)

img_dir = './images/'
def display_frames_as_gif(frames, gif_filename='./none.gif'):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
                                    
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save(img_dir+gif_filename, writer='imagemagick', fps=30)

def output_frames(frames):
    for frame in frames:
        print(frame)

class FFN(nn.Module):
    def __init__(self, in_dim, out_dim, is_actor=True):
        super(FFN, self).__init__()
        self.hidden_dim = 32
        self.is_actor = is_actor

        self.layers = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                #nn.Linear(self.hidden_dim, self.hidden_dim),
                #nn.ReLU(),
                nn.Linear(self.hidden_dim, out_dim),
                #nn.Dropout(0.5),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, observation):
        x = self.layers(observation)
        if self.is_actor:
            x = self.softmax(x)
        return x

class PPO(object):
    def __init__(self, env):
        self.n_updates = 10
        self.actor_lr = 0.005
        self.critic_lr = 0.01
        self.gamma = 0.99
        self.clip = 0.2

        self.env = env
        self.obs_dim = env.observation_space.shape[0]# 观测/状态空间
        self.act_dim = env.action_space.n # 动作空间

        self.actor = FFN(self.obs_dim, self.act_dim)
        self.critic = FFN(self.obs_dim, 1, is_actor=False)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)


    def train(self):
        print('*'*8, 'start learning...')
        self.actor.train()
        self.critic.train()
        max_rewards = 0
        for eps in range(30000):
            batch_observation, batch_action, batch_reward, batch_log_probs, batch_value = self.rollout(self.n_updates)
            batch_observation = torch.Tensor(batch_observation).view(-1, self.obs_dim)
            batch_action = torch.LongTensor(batch_action)
            batch_reward_togo = self.compute_reward_togo(batch_reward)
            batch_reward_togo = torch.Tensor(batch_reward_togo)
            batch_reward_togo = (batch_reward_togo - torch.mean(batch_reward_togo)) / (torch.std(batch_reward_togo) + 1e-10)
            batch_log_probs = torch.stack(batch_log_probs, dim=0).squeeze() 
            batch_value = torch.stack(batch_value, dim=0).squeeze() 
            A_k = batch_reward_togo.detach() - batch_value.detach()
            A_k = (A_k - torch.mean(A_k)) / (torch.std(A_k) + 1-10)

            for _ in range(5):
                probs = self.actor(batch_observation)
                cur_batch_log_probs = torch.log(torch.gather(probs, dim=1, index=batch_action.view(-1, 1))).squeeze()
                ratios = torch.exp(cur_batch_log_probs - batch_log_probs.detach())
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                actor_loss = torch.mean(-torch.minimum(surr1, surr2))
                critic_loss = torch.nn.MSELoss()(self.critic(batch_observation), batch_reward_togo.view(-1, 1)).sum() / self.n_updates
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optim.step()
                self.critic_optim.step()
                print(sum(map(sum, batch_reward)) / self.n_updates, actor_loss.detach().item(), critic_loss.detach().item())

        torch.save(self.critic, "models/ppo_critic")

        self.rollout(1, render=True)


    def get_action(self, obs):
        obs = obs.view(-1)
        probs = self.actor(obs)
        # 注意：在test阶段，action是确定的，即就是上面的"probs"，但train阶段actor应该是去尽量『探索』各种actions，为了模拟这个探索的过程，
        # 我们借助概率分布来模拟
        dist = torch.distributions.Categorical(probs) # 以概率probs形成一个类别分布
        action = dist.sample([1]) # 指定生成样本的维度 # [1]
        log_prob = dist.log_prob(action) # [1]

        value = self.critic(obs)
        return action.detach().item(), log_prob.detach(), value.detach()

    def rollout(self, times, render=False):
        batch_observation = []
        batch_log_probs = []
        batch_action = []
        batch_reward = []
        batch_value = []
        
        frames = []
        for batch in range(times):
            reward_list = []
            observation = self.env.reset()
            for t in range(100000):
                if render:
                    frames.append(self.env.render())
                batch_observation.append(observation)
                action, log_prob, value = self.get_action(torch.Tensor(observation))
                batch_log_probs.append(log_prob)
                observation, reward, done, info = self.env.step(action)
                batch_action.append(action)
                reward_list.append(reward)
                batch_value.append(value)
                if done:
                    #if reward == 1:
                        #self.env.render()
                    break
            batch_reward.append(reward_list)
        if render:
            #display_frames_as_gif(frames, "test.gif")
            output_frames(frames)

        return batch_observation, batch_action, batch_reward, batch_log_probs, batch_value

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
    #env = gym.make("Gomoku6x6-v0")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    ppo = PPO(env)
    ppo.train()
