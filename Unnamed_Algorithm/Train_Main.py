"""
该文件用于执行模型的训练
"""
from Unnamed_Algorithm.Network import *
from Unnamed_Algorithm.Environment_Interaction import *

ITERATION_NUM = 10    # 训练轮数
GAMMA = 0.95            # 衰减率[0-1]
ACTOR_LR = 0.0001       # actor网络的学习率
CRITIC_LR = 0.0001      # critic网络的学习率

class Agent:
    # 模型
    actor = None
    critic = None

    def __init__(self):
        # 模型初始化
        self.actor = Actor(ma_aims_num=MS_NUM + AIMS_NUM, node_num=NODE_NUM)
        self.critic = Critic(ma_aims_num=MS_NUM + AIMS_NUM, node_num=NODE_NUM)

    def run(self):
        """
        按照全部部署完一次，算作迭代一次
        :return:
        """
        # 算法执行
        actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        # 迭代训练

if __name__ == '__main__':

    agent = Agent()
    agent.run()