# AI-Microservices-and-Reinforce-Learning

## 项目架构

### Baselines Algorithms

用于定义对比算法

---

### Environment
用于定义环境
#### ENV_DEF.py
服务环境：

**定义**：

1. 定义了请求链集合的生成，以及边缘节点的创建方法
2. 定义用户只发生一个请求链，用户和服务器节点都包含地理位置信息（x,y）
3. 定义每一个用户有一个请求到达率`lamda`，这个值在[3-6]之间

**描述**：

1. 每一个请求链有用户生成，包含若干微服务（2-4个）和AI微服务（0-3个）
2. 每一个微服务需要使用：`cpu`、`memory`(内存)
3. 每一个AI微服务需要使用：`cpu`、`gpu`、`memory`
4. 微服务和AI微服务有一定的处理速率`alpha`


**函数**：
1. 使用函数 `get_user_request` 返回：(用户对象列表, 每个用户的有且仅有一个的请求链的集合（字典）, 判定微服务是否为AI微服务的列表)
2. 使用函数 `edge_initial` 创建边缘节点的列表

---

### Unnamed Algorithm

暂时为命名的主算法

#### network.py

定义了用于强化学习算法()

main_test.py 用于测试