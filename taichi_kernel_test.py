import numpy as np
import taichi as ti
import random

ti.init(arch=ti.vulkan)  # Try to run on GPU

final_pixels = ti.Vector.field(3, dtype=float, shape=(3,))
samples_per_pixel = 3


@ti.data_oriented
class bvh_node:
    def __init__(self, left):
        if left == 0:
            self.left = bvh_node(1)
        elif left == 1:
            self.left = 1

b = [1, 2, 4]

a = ti.field(float, shape=(3,))
a[0] = 1.0
a[1] = 3.0
a[2] = 2.0


@ti.kernel
def test():
    for i in range(1):
        for index in a:
            print(index)


shape_len = [2, 2, 4]
total_size = sum(shape_len)
shape = (total_size, 2)
shape_index = np.zeros(shape, dtype=np.int32)
a = ti.Vector.field(2, ti.i32, shape=(total_size, ))

shape_len_arr = [sum(shape_len[:i+1]) for i in range(len(shape_len))]


cat_index = 0
each_index = 0
for i in range(total_size):
    
    for j in range(len(shape_len_arr)):
        if i == shape_len_arr[j]:
            each_index = 0
            cat_index += 1

    shape_index[i, 0] = cat_index
    shape_index[i, 1] = each_index

    
    each_index+=1

print(shape_index)

a.from_numpy(shape_index)

print(a[0][1])

test()