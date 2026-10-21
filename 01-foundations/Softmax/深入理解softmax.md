# 深入理解softmax

**softmax(归一化指数函数)**:一种将任意实数向量转换为概率分布的函数。它是深度学习中多分类任务的"标配"输出函数。

简单来说就是将一组“**分数**”转换为一组“**概率**”，使其和等于1，softmax作为转换函数。


## 一、 **公式：**
### 1.1 核心公式、
一个n维实数向量x = [$x_1,x_2,……,x_n$]：
> $$
> Softmax(x_i) = {\\\frac{e^{x_i}}{\\\sum_{j=1}^n{e^{x_j}}}}
> $$

$e^{x^i}$ > 0，保证每个元素输出为正，放大数值差异
$\\\sum_{j}e^{x_j}$取所有指数之和，作为归一化因子

~~~
import numpy as np
def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x)
if __name__ == "__main__":
    logtist = [10, 2, 1000]
    print(softmax(logtist))
~~~
[ 0.  0. nan]
出现大数值，容易出现上溢。


### 1.2 稳定版公式：解决大数值上溢问题
数学证明：减去最大值并不改变结果
> $$ \\\frac{e^{{x_i}-c}}{\\\sum_{j}e^{x_j-c}} = {\\\frac{{e^x_i}•{e^{-c}}}{e^{-c}•{\\\sum_{j}e^{x^j}}}}= {\\\frac{e^{x_i}}{\\\sum_{j}e^{x_j}}}$$


~~~
def softmax_stable(x):
    x = np.asarray(x)
    x_shifted = x - np.max(x)
    p_x = np.exp(x_shifted)
    return p_x / np.sum(p_x)
if __name__ == "__main__":
    logtist = [10, 2, 1000]
    print(softmax_stable(logtist))
~~~
[0. 0. 1.]


另外，还有下溢问题：
>   logtist差距非常大，比如[0, 1，   -1000]，$e^{-1000}$作为分子会直接变为0，导致结果为0，计算机的浮点数 float64（双精度）能表示的最小正数大约是 5e-324（再往下就是 0 了）。
>
>   而 e^{-1000} ≈ 5.08e-435，比 5e-324 还小了几百个数量级，计算机表示不了这么小的数，只能把它舍入成 0.0。这种现象叫下溢（underflow）
    若是先使用softmax，再对其结果取log会出现-inf。
    print(np.log(softmax(logtist)))
    [-1.31326169 -0.31326169        -inf]
### 1.3 logsoftmax：实际工程运用较多

> $$LogSoftmax(x_i) = log({\\\frac{e^{x_i}}{\\\sum_{j}e^{x_j}}}) = x_i-log{\\\sum_{j}e^{x_j}}$$

>   使用LogSoftmax一来可以避免概率值过小导致下溢，增强数值稳定；二可以提高计算效率

~~~
def logsoftmax(x):
    x = np.asarray(x)
    x_shifted = x - np.max(x)
    return x_shifted - np.log(np.sum(np.exp(x_shifted)))
if __name__ == "__main__":
    logtist = [10, 2, 1000]
    print(logsoftmax(logtist))
~~~
[-990. -998.    0.]

### 1.4 带温度的SoftMax：通过引入“温度”参数T调整分布平滑程度：
> $$Softmax(x_i/T) = {\\\frac{e^{{x_i}/T}}{\\\sum_{j}e^{{x_j}/T}}}$$



~~~
def softmax_T(x, T = 1.0):
    x = np.asarray(x)
    x_shifted = x - np.max(x)
    p_x = np.exp(x_shifted / T)
    return p_x / np.sum(p_x)
if __name__ == "__main__":
    logtist = [10, 2, 1000]
    logtist2 = [1, 3, 7]
    print(softmax_T(logtist))
    print(softmax_T(logtist, T = 0.1))
    print(softmax_T(logtist, T = 10))
    print(softmax_T(logtist2))
    print(softmax_T(logtist2, T = 0.1))
    print(softmax_T(logtist2, T = 10))
~~~
logtist = [10, 2, 1000]
[0. 0. 1.]
[0. 0. 1.]
[1.01122149e-43 4.54371106e-44 1.00000000e+00]
logtist2 = [1, 3, 7]
[0.00242826 0.01794253 0.97962921]
[8.75651076e-27 4.24835426e-18 1.00000000e+00]
[0.24730918 0.30206411 0.45062671]


**小结：**
-   T < 1   分布更尖锐，更自信
-   T = 1   标准Softmax
-   T > 1   分布更平滑

## 二、引用
[深入理解 Softmax 函数：原理、公式与实战](https://omegazhang3.github.io/ml-blog/posts/softmax/)
[深入理解Softmax：数值稳定性、梯度传播与温度系数的工程实践](https://blog.csdn.net/weixin_29085865/article/details/168584710)