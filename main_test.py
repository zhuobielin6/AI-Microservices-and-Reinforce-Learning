import torch

# 定义概率分布
probs = torch.tensor([0.7320, 0.2680])

# 使用 torch.multinomial 根据概率分布随机选择一个索引
# 注意：这里我们只需要一个样本，所以 num_samples=1


# 由于我们的列表是 [0, 1]，我们可以直接根据索引返回对应的值
# 但由于 index 是一个张量，我们需要使用 item() 方法来获取其 Python 数值
for _ in range(10):
    index = torch.multinomial(probs, num_samples=1).item()

    # 直接打印索引（它对应于列表中的值）
    print(index)  # 输出将是 0 或 1，根据概率分布随机决定
