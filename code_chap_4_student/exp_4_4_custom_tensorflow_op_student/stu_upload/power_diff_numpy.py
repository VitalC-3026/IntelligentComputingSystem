# coding:utf-8
import numpy as np

def power_diff_numpy(input_x,input_y,input_z):
    # TODO:完成numpy实现的过程，参考实验教程示例
    x_shape = np.shape(input_x)
    y_shape = np.shape(input_y)
    print(x_shape)
    print(y_shape)
    x = np.reshape(input_x, (-1, y_shape[-1]))
    x_new_shape = np.shape(x)
    print(x_new_shape)
    y = np.reshape(input_y, (-1))
    output = []
    for i in range(x_new_shape[0]):
        out = np.power(x[i] - y, input_z)
        output.append(out)
    print(type(output))
    output = np.array(output).reshape(x_new_shape)
    print(type(output))
    return output

