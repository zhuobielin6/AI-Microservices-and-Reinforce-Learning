"""
该脚本用于实现微服务的环境交互工作
"""
from Environment.ENV import *
from Unnamed_Algorithm.network import *

torch.manual_seed(0)  # 随机数种子
PUNISHMENT_DEPLOY_FAIL = -10  # 部署失败的惩罚


class Environment_Interaction:

    def option_ms(self):
        """
        选择一个微服务进行分配
        根据当前所需的实例数情况，返回最高需求量的那个
        返回-1表示已经全部部署完毕
        :return:MaxIndex
        """
        index = np.argmax(self.ms_image)
        if self.ms_image[index] == 0:
            return -1
        return index

    def allocate_resources(self, index, state, action):
        """
        给某一个服务分配资源，若不够用则返回-1
        :param index: 选择的微服务类型
        :param state: 状态
        :param action: 行动，即选择部署的节点
        :return:
        """
        # 分离得到部署情况和资源情况
        deploy = get_deploy(state)
        resource = get_resource(state)

        # 需要消耗的资源情况
        cpu = self.ms_aims[index].get_cpu()
        memory = self.ms_aims[index].get_memory()
        gpu = self.ms_aims[index].get_gpu()

        # 当前的资源状况
        # print(resource)
        now_cpu = resource[NODE_NUM: NODE_NUM * 2][action]
        now_memory = resource[NODE_NUM * 5:][action]
        now_gpu = resource[NODE_NUM * 3: NODE_NUM * 4][action]

        if now_cpu < cpu or now_memory < memory or now_gpu < gpu:
            print(
                f'资源不够不能分配！当前CPU：{now_cpu}、GPU：{now_gpu}、内存：{now_memory}，需要的资源：CPU：{cpu}、GPU：{gpu}、内存：{memory}')
            return False

        # 可以分配资源，则开始分配
        ## 分配实例数
        # print(state)
        self.ms_image[index] -= 1
        deploy[index][action] += 1
        ## 配平相应的资源
        # cpu分配
        resource[NODE_NUM: NODE_NUM * 2][action] -= 1
        resource[:NODE_NUM][action] += 1
        # gpu分配
        resource[NODE_NUM * 3: NODE_NUM * 4][action] -= 1
        resource[NODE_NUM * 2: NODE_NUM * 3][action] += 1
        # 内存分配
        resource[NODE_NUM * 5:][action] -= 1
        resource[NODE_NUM * 4: NODE_NUM * 5][action] += 1

        return True

    def get_next_state(self, state, action_probabilities):
        """
        根据当前状态和行动列表执行下一步行动得到新的状态
        :param state:
        :param action_probabilities:
        :return:
        """
        ## 找到要部署的服务
        index = self.option_ms()
        if index == -1:
            print("已部署完成")
            return None
        print("待分配实例数情况：", self.ms_image)
        print(f"选择第{index}个服务进行部署")

        # 找到行动，即按指定概率随机选择一个服务节点
        action = torch.multinomial(action_probabilities[index], num_samples=1).item()
        print("当前actor产生的行动（每个微服务选择每一个服务器节点部署的概率分布）如下：\n",action_probabilities)
        print(f"根据概率{action_probabilities[index]}选择了行动{action}，作为部署节点")
        ## 资源分配
        # 初始化
        next_state = state.copy()
        # 分配失败
        if not self.allocate_resources(index, next_state, action):
            print("分配失败")

        # 分配成功，返回下一个状态
        return next_state

    def get_reward(self):
        """
        针对某个状态和行动返回奖励
        :return:
        """

    def refresh(self):
        """
        刷新，即重新分配实例数
        :return: None
        """
        self.ms_image = self.old_ms_image.copy()

    def __init__(self, ms_image, ms, aims):
        """
        初始化时，需要给实例数
        :param ms_image: 实例数镜像
        """
        # 初始化资源镜像
        self.old_ms_image = ms_image.copy()
        self.ms_image = self.old_ms_image.copy()

        # 初始化服务
        self.ms_aims = ms + aims


if __name__ == '__main__':
    # 制作一个待分配实例数
    ms = ms_initial()
    aims = aims_initial()
    user = user_initial()
    node_list = edge_initial()
    users, user_list, marke = get_user_request(user)
    ms_image = get_ms_image(ms, aims, users, user_list, marke)

    # 初始化环境
    # print(ms_image, type(ms_image))
    env = Environment_Interaction(ms_image, ms, aims)
    # print(env.option_ms())

    # 生成一个初始状态
    state = initial_state()

    # 模型
    actor = Actor(MA_AIMS_NUM, NODE_NUM)
    critic = Critic(MA_AIMS_NUM, NODE_NUM)

    # 测试部分
    print("每一种微服务资源所需情况如下：")
    for i,s in enumerate(ms + aims):
        print(f"服务{i}，所需资源:CPU:{s.get_cpu()}, GPU:{s.get_gpu()},内存：{s.get_memory()}")
    for _ in range(5):
        print("当前状态:\n", state)
        print("当前部署情况:\n", get_deploy(state))
        print("当前资源情况:\n", get_resource(state))
        action_list = actor(state)
        next_state = env.get_next_state(state, action_list)
        state = next_state
