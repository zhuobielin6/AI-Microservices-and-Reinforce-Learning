import torch
import torch.nn as nn
import torch.nn.functional as F
from Environment.ENV import *

class Actor(nn.Module):
    def __init__(self, ma_aims_num, node_num):
        super(Actor, self).__init__()
        self.ma_aims_num = ma_aims_num
        self.node_num = node_num

        # 输入维度规模
        input_size = ma_aims_num * node_num + ma_aims_num * node_num ** 2 + 3 * 2 * node_num
        # 输出维度
        output_size = ma_aims_num * node_num

        # LSTM层
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, output_size)

        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, inputs):
        # 转化输入为torch.Tensor对象
        if not isinstance(inputs, torch.Tensor):
            x = torch.tensor(inputs, dtype=torch.float32)
        else:
            x = inputs

        # 输入需要增加时间步维度和批量维度，形状变为 (batch_size=1, seq_len=1, input_size)
        x = x.unsqueeze(0).unsqueeze(0)

        # LSTM 前向传播
        lstm_out, _ = self.lstm(x)  # 输出形状为 (batch_size=1, seq_len=1, hidden_size=128)
        x = lstm_out.squeeze(0).squeeze(0)  # 取出张量，仅保留 (hidden_size=128)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # 形状调整为 (MA_AIMS_NUM, NODE_NUM)
        x = x.view(self.ma_aims_num, self.node_num)

        # Softmax 处理，每行表示概率分布
        probabilities = F.softmax(x, dim=1)

        return probabilities


class Critic(nn.Module):
    def __init__(self, ma_aims_num, node_num):
        super(Critic, self).__init__()
        self.ma_aims_num = ma_aims_num
        self.node_num = node_num

        # 输入维度
        input_size = ma_aims_num * node_num + ma_aims_num * node_num ** 2 + 3 * 2 * node_num

        # LSTM层
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 1)  # 输出为单个评价值

        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, inputs):
        # 转化输入为torch.Tensor对象
        if not isinstance(inputs, torch.Tensor):
            x = torch.tensor(inputs, dtype=torch.float32)
        else:
            x = inputs

        # 输入需要增加时间步维度和批量维度，形状变为 (batch_size=1, seq_len=1, input_size)
        x = x.unsqueeze(0).unsqueeze(0)

        # LSTM 前向传播
        lstm_out, _ = self.lstm(x)  # 输出形状为 (batch_size=1, seq_len=1, hidden_size=128)
        x = lstm_out.squeeze(0).squeeze(0)  # 取出张量，仅保留 (hidden_size=128)

        # 全连接层
        x = F.relu(self.fc1(x))
        value = self.fc2(x)

        return value


# 示例代码
if __name__ == "__main__":
    MA_AIMS_NUM = MS_NUM + AIMS_NUM

    # 输入数据示例
    example_input = initial_state()  # 创建一个输入数据
    print("输入状态数据如下")
    print(example_input)

    # 初始化 Actor 和 Critic 网络
    actor = Actor(MA_AIMS_NUM, NODE_NUM)
    critic = Critic(MA_AIMS_NUM, NODE_NUM)

    # 前向传播
    action_probabilities = actor(example_input)
    action_value = critic(example_input)

    print("actor网络的行动输入如下所示(Actor Output):")
    print(action_probabilities)
    print("\ncritic网络对该行动给出的评价 (Critic Output):")
    print(action_value)
"""
以下为一个输出示例：
actor网络的行动输入如下所示(Actor Output):
tensor([[0.4647, 0.5353],
        [0.4336, 0.5664],
        [0.5657, 0.4343],
        [0.5281, 0.4719],
        [0.4779, 0.5221],
        [0.4355, 0.5645],
        [0.5096, 0.4904],
        [0.4376, 0.5624]], grad_fn=<SoftmaxBackward0>)

critic网络对该行动给出的评价 (Critic Output):
tensor([0.3287], grad_fn=<ViewBackward0>)
"""