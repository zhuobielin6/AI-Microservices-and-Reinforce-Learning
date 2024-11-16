import random
import math
import matplotlib.pyplot as plt
import numpy as np

MS_NUM = 10  # 微服务数
AIMS_NUM = 5  # AI微服务数
NODE_NUM = 10  # 边缘节点数
USER_NUM = 15  # 用户数


class MS:
    '''
    基础微服务拥有两种资源类型
    '''

    def __init__(self, id) -> None:
        self.id = id
        self.alpha = random.randint(2, 4)
        self.cpu = random.uniform(1, 2)
        self.memory = random.randint(10, 15)

    def get_alpha(self):
        return self.alpha

    def get_cpu(self):
        return self.cpu

    def get_memory(self):
        return self.memory


class AIMS:
    '''
    AI微服务需要三种资源类型
    AI微服务的处理速率alpha由组成它的dnn网络的处理速率决定
    这里我们采用时间估计的方式，通过估计处理AI微服务所需要的时间反向推理它的处理速率
    '''

    def __init__(self, id) -> None:
        self.id = id
        self.dnn_num = random.randint(4, 6)
        self.cpu = random.uniform(5, 8)
        self.gpu = random.uniform(1, 2)
        self.memory = random.randint(50, 80)

    def get_alpha(self):
        exe_time = 0
        for _ in range(self.dnn_num):
            dnn_alpha = random.randint(5, 7)
            exe_time += 1 / dnn_alpha
        return 1 / exe_time

    def get_cpu(self):
        return self.cpu

    def get_gpu(self):
        return self.gpu

    def get_memory(self):
        return self.memory


class EDGE_NODE:
    '''
    边缘节点，拥有位置信息，以及资源数量
    '''

    def __init__(self, id) -> None:
        self.id = id
        self.x = random.uniform(10, 100)
        self.y = random.uniform(20, 80)
        self.cpu = random.uniform(15, 25)
        self.gpu = random.uniform(0, 3)
        self.memory = random.randint(300, 400)

    def get_location(self):
        return self.x, self.y

    def get_cpu(self):
        return self.cpu

    def get_gpu(self):
        return self.gpu

    def get_memory(self):
        return self.memory


class USER:
    '''
    用户等价与服务请求
    拥有位置和流量
    '''

    def __init__(self, id) -> None:
        self.id = id
        self.lamda = random.randint(3, 6)
        self.x = random.uniform(0, 150)
        self.y = random.uniform(0, 100)

    def get_lamda(self):
        return self.lamda

    def get_location(self):
        return self.x, self.y

    def get_request(self):
        """
        创造一个含有微服务和AI微服务的请求链
        其中微服务个数范围是：2-4
        其中AI微服务个数范围是：0-3
        :return:
        """
        request_service = []
        ms_list = ms_initial(ms_num=MS_NUM) # 获取一个微服务请求集合
        aims_list = aims_initial(aims_num=AIMS_NUM) # 获取AI微服务请求集合
        num_of_MS = random.randint(2, 4)
        num_of_AIMS = random.randint(0, 3)

        for _ in range(num_of_MS):  # 随机抽取num_of_MS个微服务
            ms = random.choice(ms_list)
            request_service.append(ms)
            ms_list.remove(ms)  # 不重复抽取
        for _ in range(num_of_AIMS):
            aims = random.choice(aims_list)
            request_service.append(aims)
            aims_list.remove(aims) # 不重复抽取
        return request_service


# ms initial
# list [MS0,MS1,...]
def ms_initial(ms_num):
    ms_list = []
    for i in range(ms_num):
        ms_list.append(MS(i))
    return ms_list


# aims initial
# list [AIMS0,AIMS1,...]
def aims_initial(aims_num):
    aims_list = []
    for i in range(aims_num):
        aims_list.append(AIMS(i))
    return aims_list


# list
# [USER0,USER1,...]
def user_initial(user_num):
    user_list = []
    for i in range(user_num):
        user_list.append(USER(i))
    return user_list


# edge initial
# list[EDGE0,EDGE1,...]
def edge_initial(node_num):
    edge_node_list = []
    for i in range(node_num):
        edge_node_list.append(EDGE_NODE(i))
    return edge_node_list


def get_user_request(num_of_user):
    '''
    user_list里面存的是对应的（用户：用户所包含的请求链）
    访问元素的时候需要用user所索引

    我们假设一个用户只会产生一个服务请求
    :param num_of_user:
    :return:
    '''
    user = user_initial(num_of_user)
    user_list = {}
    for item in user:   # 每一个用户都会产生一个用户请求
        user_list[item] = item.get_request()
    return user_list


if __name__ == '__main__':

    user_list = get_user_request(USER_NUM)  # 随机生成USER_NUM个用户产生的请求链 集合
    node_list = edge_initial(NODE_NUM)  # 随机产生NODE_NUM个边缘服务器节点 列表

    for user in user_list:  # 遍历每一个请求集合（字典）即请求链
        # 这里的user其实是字典user_list对应的“键”
        for i in user_list[user]:   # 遍历用户user的请求链的每一个服务
            print(i.id, end=' ')
        print(' ')
        for i in user_list[user]:   # 遍历每一个微服务的处理速率
            if isinstance(i, MS):   # isinstance用于判断服务i是微服务还是AI微服务
                print(i.alpha, end=' ')
            else:
                print(i.get_alpha(), end=' ')
        print(' ')
        print("用户持有的服务请求到达率：", user.lamda)  # 请求到达率
        print(user_list[user], end=' ') # 请求链信息
        print(' ')

    '''
    输出用户与服务器之间的位置关系图
    '''
    x_list = []
    y_list = []
    for i in node_list:
        x, y = i.get_location() # 返回节点的横纵坐标
        x_list.append(x)
        y_list.append(y)
    plt.scatter(x_list, y_list, c='red', marker='*')
    user_x_list = []
    user_y_list = []
    for i in user_list:
        x, y = i.get_location()
        user_x_list.append(x)
        user_y_list.append(y)
    plt.scatter(user_x_list, user_y_list, c='blue')
    plt.tight_layout()
    plt.show()
