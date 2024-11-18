import numpy as np
from ENV_DEF import *

random.seed(123)


def initial_state():
    '''
    deploy_state:(MS_NUM+AIMS_NUM)*NODE_NUM
    rout:NODE_NUM*(MS_NUM+AIMS_NUM)*NODE_NUM
    :return:
    '''
    deploy_state = np.zeros(shape=(MS_NUM + AIMS_NUM, NODE_NUM))
    rout_state = np.zeros(shape=(NODE_NUM, MS_NUM + AIMS_NUM, NODE_NUM))
    CPU = np.zeros(shape=(2, NODE_NUM))
    GPU = np.zeros(shape=(2, NODE_NUM))
    Memory = np.zeros(shape=(2, NODE_NUM))
    node_list = []
    for i in range(NODE_NUM):
        edge_node = EDGE_NODE(i)
        node_list.append(edge_node)
        CPU[1][i] = edge_node.cpu  # 初始化剩余cpu资源
        GPU[1][i] = edge_node.gpu  # 初始化剩余gpu资源
        Memory[1][i] = edge_node.memory  # 初始化剩余memory资源
    deploy_state = np.reshape(deploy_state, (1, (MS_NUM + AIMS_NUM) * NODE_NUM))
    rout_state = np.reshape(rout_state, (1, NODE_NUM * (MS_NUM + AIMS_NUM) * NODE_NUM))
    CPU = np.reshape(CPU, (1, 2 * NODE_NUM))
    GPU = np.reshape(GPU, (1, 2 * NODE_NUM))
    Memory = np.reshape(Memory, (1, 2 * NODE_NUM))
    resource = np.append(CPU, GPU)
    resource = np.append(resource, Memory)
    state = np.append(deploy_state, rout_state)
    state = np.append(state, resource)
    return state


def get_deploy(state):
    deploy = state[0:(MS_NUM + AIMS_NUM) * NODE_NUM]
    deploy = np.reshape(deploy, ((MS_NUM + AIMS_NUM), NODE_NUM))
    return deploy


def get_rout(state):
    rout = []
    for i in range(NODE_NUM):
        rout_node = state[(MS_NUM + AIMS_NUM) * NODE_NUM + i * (MS_NUM + AIMS_NUM) * NODE_NUM
                          :(MS_NUM + AIMS_NUM) * NODE_NUM + (i + 1) * (MS_NUM + AIMS_NUM) * NODE_NUM]
        rout_node = np.reshape(rout_node, ((MS_NUM + AIMS_NUM), NODE_NUM))
        rout.append(rout_node)
    return rout


def get_ms_image(ms, aims, users, requests, marke):
    ms_image = np.zeros(MS_NUM + AIMS_NUM)
    ms_lamda = np.zeros(MS_NUM + AIMS_NUM)
    # request_lamda = get_user_lamda(users)
    # print(request_lamda)
    for user in users:
        lamda = user.lamda
        request = requests.get(user)
        single_marke = marke.get(user)
        for item1, item2 in zip(request, single_marke):
            if item2 == 0:
                ms_lamda[item1.id] += lamda
            else:
                ms_lamda[MS_NUM + item1.id] += lamda
    alpha_list = np.append(get_ms_alpha(ms), get_aims_alpha(aims))
    for i in range(MS_NUM + AIMS_NUM):
        rho = ms_lamda[i] / alpha_list[i]
        ms_image[i] += math.ceil(rho)
    return ms_image


def get_first_node(users, node_list):
    '''
    获得服务请求接收节点集
    :param users:
    :param node_list:
    :return:
    '''
    node = []
    for i in range(USER_NUM):
        user = users[i]
        node_idx = 0
        dis = float('inf')
        for item in node_list:
            dis = min(cal_dis_user_node(user, item), dis)
            if dis == cal_dis_user_node(user, item):
                node_idx = item.id
        node.append(node_idx)
    return node


def get_ms_node_lamda(state, users, requests, node_list):
    '''

    :param state: 一维向量
    :param users: list
    :param requests: dict
    :param node_list: list
    :return:
    '''
    ms_node_lamda = []
    first_node = get_first_node(users, node_list)
    deploy = get_deploy(state)
    for ms in range(MS_NUM):
        ms_request = []
        for item in requests:
            item
        for node in range(NODE_NUM):
            ms = []

    return ms_node_lamda


def jiechen(n):
    k = 1
    if n == 0:
        return 1
    else:
        for i in range(1, n + 1):
            k *= i
        return k


def cal_ms_delay(ms_deploy):
    '''
    :param ms_deploy: (NODE_NUM*MS_NUM)
    :param a:
    :return:
    '''
    ms_delay = []
    for i in range(NODE_NUM):
        ms_on_node_delay = []
        for j in range(MS_NUM):
            num = ms_deploy[i][j]


if __name__ == '__main__':
    state = initial_state()
    d = get_deploy(state)
    r = get_rout(state)
    print(state)
    print(d)
    print(r)

    ms = ms_initial()
    aims = aims_initial()
    user = user_initial()
    node_list = edge_initial()
    users, user_list, marke = get_user_request(user)
    print("用户集：", users)
    print("请求集：", user_list)
    print("标记集合", marke)
    for item in users:
        print("用户", item.id, "的位置：", item.x, item.y)
    for item in node_list:
        print("服务器", item.id, "的位置：", item.x, item.y)
    for item1 in users:
        for item2 in node_list:
            print("用户", item1.id, "到服务器", item2.id, "之间的距离：", cal_dis_user_node(item1, item2))
    print(get_first_node(users, node_list))
    msalpha = get_ms_alpha(ms)
    aimsalpha = get_aims_alpha(aims)
    servicelamda = get_user_lamda(user)
    print("基础微服务的处理速率：", msalpha)
    print("AI微服务的处理速率：", aimsalpha)
    print("服务请求的到达率：", servicelamda)
    for item in users:
        for i in user_list.get(item):
            print(i.id, end=' ')
        print(' ')
    ms_image = get_ms_image(ms, aims, users, user_list, marke)



