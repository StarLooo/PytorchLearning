import torch


def showTensorInfos(aTensor, name=None):
    if type(aTensor) == torch.Tensor:
        if name is not None:
            print(name, "张量的基本信息如下：")
        print("data:\n", aTensor.data)
        print("dtype:", aTensor.dtype)
        print("shape:", aTensor.shape)
        print("device:", aTensor.device)
        print("*****************************")
    else:
        print("参数中变量不是torch.Tensor类型！")


if __name__ == '__main__':
    # tensor_x = torch.empty(5, 3)
    # showTensorInfos(tensor_x, "tensor_x")

    # tensor_x = torch.rand(3, 2)
    # showTensorInfos(tensor_x)
    #
    # tensor_x = torch.zeros(4, 4, dtype=torch.float32)
    # showTensorInfos(tensor_x)
    #
    # tensor_x = torch.randn_like(tensor_x)
    # showTensorInfos(tensor_x)
    #
    # tensor_y = torch.rand(4,4)
    # tensor_z = tensor_y.add(tensor_x)
    # showTensorInfos(tensor_z)
    # showTensorInfos(tensor_y)
    #
    # tensor_z = tensor_y.add_(tensor_x)
    # showTensorInfos(tensor_z)
    # showTensorInfos(tensor_y)

    # tensor_y = torch.clone(tensor_x)
    # showTensorInfos(tensor_y)
    #
    # tensor_x.add_(1)
    # showTensorInfos(tensor_x, "tensor_x")
    # showTensorInfos(tensor_y, "tensor_y")

    # tensor_x = torch.rand(4,4)
    # tensor_y = tensor_x.t()
    # showTensorInfos(tensor_x, "tensor_x")
    # showTensorInfos(tensor_y,"tensor_y")

    # tensor_x = torch.ones(2,2,requires_grad=True)
    # print(tensor_x)
    # print(tensor_x.grad_fn)
    #
    # tensor_y = tensor_x + 2
    # print(tensor_y)
    # print(tensor_y.grad_fn)
    #
    # print(tensor_x.is_leaf, tensor_y.is_leaf)
    #
    # tensor_z = tensor_y * tensor_y * 3
    # tensor_out = tensor_z.mean()
    # print(tensor_z)
    # print(tensor_out)
    #
    # print("***********************************")
    #
    # tensor_out.backward()
    # print(tensor_x.grad)
    #
    # # 再来反向传播一次，注意grad是累加的
    # tensor_out2 = tensor_x.sum()
    # print(tensor_out2)
    # tensor_out2.backward()
    # print(tensor_x.grad)
    #
    # tensor_out3 = tensor_x.sum()
    # tensor_x.grad.data.zero_()
    # tensor_out3.backward()
    # print(tensor_x.grad)

    # tensor_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    # tensor_y = 2 * tensor_x
    # tensor_z = tensor_y
    #
    # print(tensor_z)
    #
    # tensor_weight = torch.tensor([[1.0, 0.1], [0.01, 0.001]])
    #
    # tensor_z.backward(tensor_weight)
    #
    # print(tensor_x.grad)

    # x = torch.tensor(1.0, requires_grad=True)
    # y1 = x ** 2
    # with torch.no_grad():
    #     y2 = x ** 3
    # y3 = y1 + y2
    #
    # print(x.requires_grad)  # True
    # print(y1, y1.requires_grad)  # True
    # print(y2, y2.requires_grad)  # False
    # print(y3, y3.requires_grad)  # True
    #
    # y3.backward()
    # print(x.grad)

    x = torch.ones(1, requires_grad=True)

    print(x.data)  # x.data还是一个tensor
    print(x.data.requires_grad)  # 但是x.data已经是独立于计算图之外

    y = 2 * x
    x.data *= 100  # 只改变了x.data值，不会记录在计算图，所以不会影响梯度传播

    y.backward()
    print(x)  # 更改x.data的值也会影响x的值
    print(y)  # 更改x.data的值不会影响y的值
    print(x.grad)
