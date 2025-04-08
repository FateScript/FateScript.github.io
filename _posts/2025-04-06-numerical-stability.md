---
layout: post
title: 那些年，我们没想过的数值稳定算法
date: 2025-04-06 11:59:00-0400
description: 数值稳定：管理误差的艺术
categories: deep-learning engineering
tags: [code,]
giscus_comments: true
enable_math: true
toc: true
---

深度学习模型的训练在本质上是通过一系列复杂的数值计算组合去逼近一个极度复杂的函数$f$，而机器本身对于数值的表达就带有精度上的误差。真实的世界往往会遭遇这样的情况：一个微小的浮点误差，导致了最终的梯度爆炸或消失，抑或是模型无法收敛，而这让找寻原因变得异常困难。

对于大部分炼丹师来说，结构上的更改和巧思显然令人着迷，但是本文我们不讨论为什么attention要除以 $\sqrt{d_k}$，抑或是GPT为什么用pre-norm而不是post-norm这种，这些属于模型结构上的巧思，而这篇文章关注的是数学和工程层面的技巧。数值分析之父 James H. Wilkinson曾经表达过这样的观点：“数值计算的主要挑战在于如何管理误差的传播(propagation of rounding errors)”。而这些管理技巧又可以分成两个部分，即：

- 使得数值计算的方法给出的结果更接近真实结果
- 在计算结果不能更优的基础上，防止误差进一步扩散


### 前置概念

为了防止你在后面看的太过云里雾里，脑海中想象不到在底层到底发生了什么，我们稍微温习一下[IEEE754标准](https://en.wikipedia.org/wiki/IEEE_754)以及介绍一下衍生的数值问题场景。

IEEE754标准在表示浮点数的时候，主要分为三个部分：符号位、指数位和尾数位。我们用一个32位的浮点数如何表示79进行举例：
{% include figure.html path="assets/blog/float_num.jpg" class="img-fluid rounded z-depth-1" %}​
注意这里exponent是8位的，所以指数对应的数值是6 + 127 = 133，是规格化的浮点数。非规格化的数据我们暂不讨论，此处只是快速回顾一下浮点数的表示方法。

不同精度的浮点数只是exponent和mantissa的位数不同，比如double/float64就是11位exponent和52位mantissa，float16是5位exponent和10位mantissa，bfloat16是8位exponent和7位mantissa，而float8有两种表示方法：e4m3（4位exponent，3位mantissa）和e5m2（5位exponent，2位mantissa）。

了解了浮点数如何被表征，我们就可以讨论一下数值计算中常见的问题了，基本上可以分为三类：

1. **overflow**：很大的数据被round成了 $\infty$ 或者 $-\infty$ ，比如在IEEE754标准里面double的最大数值在1e308这个量级（可以用 `numpy.finfo(np.float64).max` 校验），如果你在python里用 1e308 + 1e308，得到的结果就是inf了
2. **underflow**：接近0的结果被round成了0，比如double的最小的正数是 $2^{-1074}$ ，这个-1074来源于 -1023(11 bit exp) - 51(52 bit mantissa)，大约为5e-324，所以如果在python里使用`9e-324 - 8e-324`，得到的结果就是0了，这里就是发生了underflow
3. **loss of precision**：因为浮点数的表示方法在数轴上是不均匀的（或者说是分段均匀的），所以天然存在一个问题：只能近似表示某些数据，这就涉及到近似表示带来的精度损失。比如在python里面，计算0.1 + 0.2，得到的结果会是0.30000000000000004，0.1 + 0.2 == 0.3返回的结果也会是False，也就是发生了精度损失。如果你进一步用 `struct` 查看hex数值，0.1 + 0.2的结果是0x3FD3333333333334，而0.3则是0x3FD3333333333333（这个数值在数轴上离0.3更近）

### 数值稳定的解法

我有一位做HPC（high performance computing）的朋友总结过hpc领域提速的两板斧：**减少计算量**（比如卷积的[WinoGrad算法](https://arxiv.org/pdf/1509.09308)）、**减少IO**（比如[flash attention](https://arxiv.org/pdf/2205.14135)）。对应的，为了提高算法的数值稳定性，也有一些基本的解决套路，总体上，可以归纳为下面四类策略：

1. **重写数学公式**
    
    很多时候，数学公式在理论上是等价的，但在数值计算中可能存在极大的稳定性差异。我们在下一个部分给出了除法运算的例子，就是典型的重写公式（改变运算的组合顺序），本质上不改变算法逻辑。
    
2. **使用其他算法**
    
    有些算法虽然数学上是等价的，但在实际计算中表现差异很大。例如在求方差的时候，既可以按照方差的定义先求期望，再求方差；也可以利用 $\text{Var}(X) = \text{E}(X^2) - \text{E}(X)^2$ 求方差。我们会在下一部分的normalization中对这两种方法进行详细的分析，这里我们只需要记住：不同的算法会有不同的数值稳定性。
    
3. **提高精度或改变数值类型** 
    
    默认情况下，模型使用的是 `float32` / `float16` 来进行训练，但在关键节点上，特别是梯度累加、参数更新等环节，如果使用低精度可能会导致误差误差累积甚至训练不收敛。所以有时候为了达到更好的稳定性，经常会在某些模块中采用更高的精度，或者临时将变量转换成高精度计算再转换回来。比如在混合精度训练中，往往会做loss scaling，或者在某些操作上autocast到 `float32` 以保证训练稳定。
    
4. **限制输入范围**
    
    这个方法应该是短期walk around的时候最常用的方法，比如定位到出现出现不稳定的具体算子，然后对输入或者输出做一下clip，或者加一个epsilon，类似clip(x, min=1e-5) 或者 x = x + 1e-5这种写法。通常对于一些除以极小值或者log一个极小值的case能起到效果。


### 那些年，我们错过的算子

#### div forward

相信很多人很难理解为什么除法会存在数值稳定性的问题（因为这个实现基本来自硬件的指令），但是实际上在深度学习框架里面，数据的类型是很多样的，仅仅是data type，就衍生出来了[accumulate type](https://github.com/pytorch/pytorch/blob/c65de03196ae3dbeb67ef38d43c4639b85a60ce4/aten/src/ATen/AccumulateType.h#L16)、saclar type（单个的数值，比如tensor + 5，5就是scalar。scalar在运算的时候会[转成tensor](https://github.com/pytorch/pytorch/blob/c65de03196ae3dbeb67ef38d43c4639b85a60ce4/aten/src/ATen/ScalarOps.h#L31)）和promote type等概念，而除法的不稳定性，就来自于promote type。

promote type是指两个不同类型的tensor（包含scalar，因为scalar可以隐式转成tensor）运算之后的结果应该是什么类型的。pytorch在进行数值计算（比如加减乘除）的时候，会对dtype采用下面的一个promote逻辑（参考 [doc](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)）：

1. 如果scalar或者zero-dim tensor（zero dim tensor就是在scalar上套了一层tensor，比如torch.tensor(1.2)就是一个zero-dim tensor）比和它运算的tensor的精度高一个层级（complex > floating > integral > boolean），那么结果的数据类型就是scalar的类型。比如int tensor 和浮点数做加法，结果就是float tensor；float tensor和complex做乘法，结果是complex tensor。
2. 如果scalar或者zero-dim tensor和参与运算的tensor是在一个精度层级下（比如都是float，但是一个是float32，一个是float16），那么返回的结果就是参与运算的tensor的dtype。比如float16的tensor和一个浮点数做加法，结果是float16的tensor；int16的tensor和python int做加法，结果是int16的tensor。

python中的浮点数的精度是double；int则是通过大整数算法，以30bit（64bit系统）或者15bit（32bit系统）为一节的变长数组[实现](https://github.com/python/cpython/blob/e7980ba233bcbdb811e96bd5003c7d51a4e25155/Include/cpython/longintrepr.h#L64-L91)。对于越界的int数据，torch会通过抛出异常进行制止（试试  `torch.tensor(1 << 63)` ）。但是，浮点数和低精度的tensor的运算就没有那么直观了，比如下面的code改编自[torch pr 41446](https://github.com/pytorch/pytorch/pull/41446)，不考虑promote type的情况下，很难理解为什么结果会是0。

```python
x = torch.tensor([3388.]).half()
scale = 524288.0
print(x.div(scale))  # tensor([0.])
```

当然上面的issue已经被修复了，修复的方法就是把`div(scale)` 换成 `mul(1 / scale)`。

<br>

#### div backward

对于除法，除了forward过程中的promote type之外，在反向传播的过程也会引入数值稳定的问题。有趣的是，这个问题在[tensorflow](https://github.com/tensorflow/tensorflow/pull/6562)和[pytorch](https://github.com/pytorch/pytorch/issues/43414)里面都有对应的pr和issue讨论，而且早期的实现都是不够数值稳定的版本。

对于除法运算 x / y，自动求导的结果是 $- \frac {x} {y^2}$ ，这也导致了早期的实现都是  `- x / (y * y)` 的形式，但是我们考虑接近0的数值y（比如1e-8）， y * y的结果往往很有可能出现underflow的问题；而对于比较大的数值y，y * y很有可能overflow。

而如果先计算 x / y，则这个中间结果通常处在一个更良好的范围里，所以对于除法的反传函数，  `- x / y / y`  通常是一个更加数值稳定的写法。这也是pytorch官方的[实现方式](https://github.com/pytorch/pytorch/blob/330c9577a3ce880c49475ca79e517b8741ff225b/torch/csrc/autograd/FunctionsManual.cpp#L648)。

如果想要这对两个写法有更直观的理解，可以试着跑一下下面的code

```python
a, b = 1.0, 1e-8

print(a / (b * b))  # 9999999999999998.0
print(a / b / b)  # 1e+16
```
<br>

#### Prod

prod是一个tensor中的数值的累乘，也就是 $y = \prod_{i=1} x_i$，对于这个累乘中的任何一个元素 $x_i$ 求导，结果就是 $\text{grad} *  y / x_i$ ，但是因为y是累乘结果，可能会存在非常大或者非常小的情况，因此和div一样，有underflow和overflow的风险，因此一个比较数值稳定的写法应该是 $\text{grad} * (y / x_i)$，这样要比第一种写法要更加数值稳定一些，这也是torch[官方的实现方式](https://github.com/pytorch/pytorch/blob/330c9577a3ce880c49475ca79e517b8741ff225b/torch/csrc/autograd/FunctionsManual.cpp#L825-L826) 。

<br>

#### Range

range通常给出起始数值start、结束数值end，还有步长step，因为首先要确定range的size，这里就会涉及到数值稳定问题，[这个torch commit](https://github.com/pytorch/pytorch/commit/33cc71dc55db073ba46b065e24cff0d26156376f)解决的就是这个问题，对应下面的两个写法：

```python
def range_size_a(xmin, xmax, step):
    return int((xmax - xmin) / step + 1)

def range_size_b(xmin, xmax, step):
    return int((xmax / step - xmin / step) + 1)
```

考虑到浮点数的加减法经常会涉及到精度损失，借助精度损失的经典案例0.1 + 0.2 不等于 0.3，我们可以轻松构造出下面的case：

```python
start, end, step = 0.1, 0.3, 0.1
print(range_size_a(start, end, step))  # 3
print(range_size_b(start, end, step))  # 2
```

毫无疑问，方法b会比方法a更加数值稳定。

<br>

#### 线性插值

线性插值 (Linear Interpolation，LERP)是计算两个数之间某个比例值的算法，在深度学习里面，比如数据增强里的[mix-up](https://arxiv.org/abs/1710.09412)、模型[权重融合](https://github.com/arcee-ai/mergekit/blob/09bbb0ae282c6356567f05fe15a28055b9dc9390/mergekit/merge_methods/slerp.py#L94-L97)、momentum update里是比较常见的一种方法，LERP函数的表达式是

$$
\text{lerp}(a, b, t) = (1 - t) \cdot a + t \cdot b
$$

其中a是起始值，b是结束值，t是插值因子（在0到1之间）。

或许你看到这里会有一些疑问，这么简单的式子，感觉怎么写都不会有数值稳定性问题，但是实际上上面数学表达式的写法很容易导致运算过程不满足单调性。单调性是指当a大于（或小于）b的时候，那么更大的t插值出来的结果应该更大（或更小）。

下面的例子就很好地表明采用数学表达式写法的问题，事实上也是[torch pr 18871](https://github.com/pytorch/pytorch/pull/18871) 尝试修复的问题。

```python
A, B = 4000.0, 4000.0
t = 0.4247583667749129  # float.fromhex("0x1.b2f3db7800a39p-2")
print((1 - t) * A + t * B)   # 4000.0000000000005
```

解决方法也很简单，将函数换成分段表示：

$$
\text{lerp}(A, B, t) =\begin{cases} A + (B - A) \times t, & \text{if } t < 0.5 \\B - (B - A) \times (1 - t), & \text{otherwise}\end{cases}
$$

这个写法的精妙在于：

1. 端点匹配。也就是在t=0和t=1的时候可以取到A和B的值。如果A和B两个浮点数很接近，那么B-A有可能会underflow到0；或者A和B相差很大，B-A结果容易等于B（或者A），都不可能做到恰好取到A和B的值。
2. 当A和B相等的时候，显然lerp的结果始终为固定值。
3. 保证单调性。毫无疑问这个函数是分段单调的，关键在于在连接处是否是单调的。考虑刚刚好比0.5小的浮点数s（在python中可以用`0.5 - math.ulp(0.5) / 2` 得到这个值），对于任意正浮点数u，我们有 $u \times s < u / 2$ 恒成立，所以这个函数是可以保证单调性的。

看到lerp的实现，很容易让人联想到二分查找里求median的时候的[数值问题](https://en.wikipedia.org/wiki/Binary_search#Implementation_issues)，也就是将 `median = (high + low) / 2` 改成 `median = low + (high - low) / 2` 。torch也有[commit](https://github.com/pytorch/pytorch/commit/56840f0a81e4460089740d50d3768f37e79a17fc)修过类似的问题。

<br>

#### normalization

在normalization中，求数据的均值和方差是非常基础的操作。我相信大部分人都会觉得求均值和方差就是简单套用下面的公式：

$$
\begin{aligned}
\text{E}(X) &= \frac{1}{n} \sum_{i=1}^{n} x_i \\
\text{Var}(X) &= \frac{1}{n} \sum_{i=1}^{n} (x_i - \text{E}(X))^2 \\
\text{Var}(X) &= \text{E}(X^2) - \text{E}(X)^2 = \frac{1}{n} \left( \sum_{i=1}^{n} x_i^2 - \frac{\left( \sum_{i=1}^{n} x_i \right)^2}{n} \right)
\end{aligned}
$$

在求方差的时候，如果采用标准的方差定义，需要先确定数据的均值，这样的话，需要遍历数据两次（two-pass）：第一次计算出均值，第二次计算出方差，这样对IO不是很友好。如果采用方法2计算方差，只需要遍历一次数据且能够在线更新（online update），但是这样会有非常明显的数值稳定问题，一个典型的case就是在小方差大均值的数据。

```python
x = [1e8 + 1, 1e8 + 2, 1e8 + 3]
n = len(x)
mean = sum(x) / n
sum_x = sum(x)
sum_x_square = sum(d ** 2 for d in x)

unstable_var = (sum_x_square - (sum_x ** 2) / n) / n
print(unstable_var)    # 0.0
stable_var = sum((d - mean) ** 2 for d in x) / n
print(stable_var)    # 0.6666666666666666
```

明明有方差，但是方差却被算成了0，核心原因就是非常大的数据做乘法之后很容易丢失精度。

当然，还是有只需要一次遍历（one-pass）且数值稳定的算法的，就是[welford](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)算法。

既然是one-pass的在线更新，关键在于定义更新的方法。在第n次更新的时候，welford采取下面的做法（mean和M2初始化为0）：

$$
\begin{aligned}
\delta &= x_n - \text{mean}_{n-1} \\
\text{mean}_n &= \text{mean}_{n-1} + \frac{\delta}{n} \\
\text{M2}_n &= \text{M2}_{n-1} + \delta \times (x - \text{mean}_n) \\
\text{Var}(X_n) &= \frac{\text{M2}_n}{n}
\end{aligned}
$$

这个做法除了保证了方差的数值稳定性之外，在均值上的稳定性也更好。如果按照原始算法，每次都维护一个sum，在数据样本n变大之后，sum会越来越大，sum和 $x_n$之间的差值也越大，精度的损失也越多。

<br>

#### log1p

log1p(x)计算的是log(1 + x)的数值，这里主要是精度损失问题，比如1加上一个非常小的数值，1+x很容易被舍入为1，再叠加上log操作，会直接计算为0。而在 $(0, 1]$ 这个区间里面做log运算也是非常危险的：因为输入的变化可能非常微小，但是输出的变化却是在剧烈的震荡（导数说明了一切）。

下面的例子展示了log1p的实现和原始实现的稳定性差异：
```python
import math

x = 1e-10
print(math.log(1 + x))   # 1.000000082690371e-10
print(math.log1p(x))     # 9.999999999500001e-11
```

log1p算是一个广为人知的专为数值稳定实现的算子，我们参考[glibc的实现](https://github.com/lattera/glibc/blob/895ef79e04a953cac1493863bcae29ad85657ee1/sysdeps/ieee754/dbl-64/s_log1p.c#L16)简单解释一下，感兴趣的同学可以自己点进去看源码。

首先考虑到IEEE754对于浮点数的表示方式，可以把1+x表示成如下形式：

$$
1 + x = 2^k (1 + f), \quad \text{where}
\quad \frac{1}{\sqrt{2}} < 1+f < \sqrt{2}
$$

这样的话 $\log(1+x)$ 的结果就等价于 $k \log 2 + \log(1+f)$ ，又因为 1+f 在1附近的范围内，所以我们可以使用Taylor展开，设 $s = \frac{f}{2+f}$，则

$$
\begin{aligned}
\log(1+f) &= \log(1+s) - \log(1-s)
\newline
\log(1+s) &= s - \frac{s^2}{2} + \frac{s^3}{3} - \frac{s^4}{4} + \dots
\newline
\log(1-s) &= -s - \frac{s^2}{2} - \frac{s^3}{3} - \frac{s^4}{4} + \dots
\newline
\log(1+f) &= 2s + \frac{2}{3} s^3 + \frac{2}{5} s^5 + \frac{2}{7} s^7 + \dots = 2s + s \cdot R(s)
\end{aligned}
$$

其中， $R(s) = \frac{2}{3} s^2 + \frac{2}{5} s^4 + \frac{2}{7} s^6 + \dots$ ，可以用[Remez算法](https://en.wikipedia.org/wiki/Remez_algorithm)做近似估计： $R(s) \approx Lp_1 s^2 + Lp_2 s^4 + Lp_3 s^6 + Lp_4 s^8 + Lp_5 s^{10} + Lp_6 s^{12} + Lp_7 s^{14}$ ，为了把rounding error控制在 $2^{-58.45}$ 以下，可以估算出来[具体的Lp数值](https://github.com/lattera/glibc/blob/895ef79e04a953cac1493863bcae29ad85657ee1/sysdeps/ieee754/dbl-64/s_log1p.c#L92-L98)。

当然，如果觉得这个算法过于复杂，其实也可以简单利用 $ x \to 0 $ 时的Taylor展开做近似（参考[博客](https://www.johndcook.com/blog/cpp_log_one_plus_x/)）。

$$
\ln(1 + x) = \sum_{n=1}^{\infty} (-1)^{n+1} \frac{x^n}{n} = x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4} + \frac{x^5}{5} - \cdots
$$

pytorch针对MPS backend，在float32下的实现就是做前两项的近似（具体参考[torch实现](https://github.com/pytorch/pytorch/blob/68dfd44e50f59c53698a24985039a27351862963/c10/metal/special_math.h#L604)）。我们也可以沿着这个思路写一个python版本。因为python里面所有的浮点数内部的表示都是double，double的eps value的量级大概在1e-16，所以如果仍然估计到 $x^2$，比较的数值大概在1e-6 （因为 $(1e-6)^3  < 1e-16$ ）

```python
import math

def log1p(x: float) -> float:
    if abs(x) > 1e-6:
        return math.log(1.0 + x)

    return (-0.5 * x + 1.0) * x
```

了解了log1p的算法之后，我们很自然地就会有一个新的问题：既然log1p要比log本身更加精确，那我在计算的时候为什么不用`log1p(x - 1)`来代替`log(x)`？

原因也很简单：log1p是在x比较小(x的绝对值小于1)的情况下存在数值稳定性意义，当而x比较小的时候，x - 1操作本身就会引入精度损失，log1p再提高精度损失，结果就是`log1p(x - 1)` 和 `log(x)`的结果一样。所以没有必要多此一举。

因为log1p是在x比较小的情况下存在数值稳定意义，而概率本身就满足数值比较小的定义。考虑求一个二分类的entropy，假设概率为prob，那么另一类概率就是1-prob，此时可以用`log1p(-prob)`来实现一个更加数值稳定的版本。

<br>

#### expm1

expm1是log1p的逆函数，计算的是exp(x) - 1的值。它的数值稳定性问题主要体现在x的数值很小（接近0）的情况，此时这个数值会渐近地趋向于1+x。

下面的例子很好地展示了expm1的实现和原始实现在接近0的数值下的稳定性差异：

```python
from math import exp, expm1

x = 1e-8
print(exp(x) - 1)   # 9.99999993922529e-09
print(expm1(x))     # 1.0000000050000001e-08
```


对照log1p，我们也从[glibc实现](https://github.com/lattera/glibc/blob/895ef79e04a953cac1493863bcae29ad85657ee1/sysdeps/ieee754/dbl-64/s_expm1.c#L17)上简单讲一下这个函数的数值稳定写法的原理。

和log1p类似，首先将x表示为下面的形式：

$$
x = k \ln 2 + r, \quad \text{where} 
 \quad |r| \leq 0.5 \ln 2
$$

这样的话，我们有

$$
e^x - 1 = \begin{cases} 
2^k \cdot (e^r + 1) - 1, & k < -2 \text{ or } k > 56 \\
2^k \cdot (e^r - 1) + (2^k - 1), & o.w.
\end{cases}
$$

因为r处在一个比较窄的范围内，所以我们可以用 $ x \to 0 $ 的Taylor展开直接解决战斗：

$$
e^r - 1 = r + \frac{r^2}{2} + \frac{r^3}{6} + \dots
$$

当然，glibc中的实现是使用下面的近似

$$
\frac{r(e^r + 1)}{e^r - 1} = 2 + \frac{r^2}{6} - \frac{r^4}{360} + \dots  = 2 + \frac{r^2}{6} \cdot R_1(r^2)
$$

之后再通过Remez算法做近似估计，把rounding error控制在  $2^{-61}$ 之下，求出具体的[Q值](https://github.com/lattera/glibc/blob/895ef79e04a953cac1493863bcae29ad85657ee1/sysdeps/ieee754/dbl-64/s_expm1.c#L127-L131)。

当然，和log1p一样，我们也同样可以直接用Taylor展开做近似，参考[博客](https://www.johndcook.com/cpp_expm1.html)，我们给出下面的python实现：

```python
import math

def expm1(x):
    if abs(x) < 1e-5:
        return x + 0.5 * x * x

    return math.exp(x) - 1.0
```
<br>

#### softmax及其变体

softmax函数是数值稳定的经典case，几乎每一个搞深度学习的工程师/研究员都应该或多或少地知道这里面的数值稳定技巧，在[deep learning(花书)](https://www.deeplearningbook.org/)的第四章中特意介绍了它的数值稳定技巧。softmax函数的定义是：

$$
\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

如果直接使用 $x_i$ ，通常会导致 $\exp(x_i)$ 的数值过高而导致overflow，所以更加数值稳定的写法则是：先减去 $\max(x)$的数值，再做softmax操作。也就是：

$$
\mathrm{softmax}(x_i) = \frac{e^{x_i} - \max(x)}{\sum_j e^{x_j} - \max(x)}
$$

因为softmax的结果算出来是概率，所以通常为了求交叉熵，会有计算log softmax的过程，

根据上面softmax的公式，我们有

$$
\begin{aligned}
\log (\mathrm{softmax}(x_i)) &= \log \left( \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}} \right)
\newline
&= \log \left( e^{x_i - \max(x)} \right) - \log \left( \sum_j e^{x_j - \max(x)} \right) 
\newline
&= x_i - \max(x) - \log \left( \sum_j e^{x_j - \max(x)} \right)
\end{aligned}
$$

[torch对于logsoftmax实现](https://github.com/pytorch/pytorch/pull/21672/files)就采用了这种写法，但是我们也可以引入LSE（log sum exp）来简化计算。LSE的定义是：

$$
\mathrm{LSE}(x_1, \dots, x_n) = \log \left( \sum_{i=1}^{n} e^{x_i} \right)
$$

当然，类比softmax，LSE也有一个数值稳定的写法，也就是

$$
\mathrm{LSE}(x_1, \dots, x_n) =  \max(x) + \log \left( \sum_{i=1}^{n} e^{x_i - \max(x)} \right)
$$

对log-softmax应用LSE trick，就可以把表达式简化为：

$$
\log \mathrm{softmax}(x_i) = x_i - \mathrm{LSE}(x_1, \dots, x_n)
$$

LSE trick的使用非常广泛，涉及到softmax的简化计算时，通常会使用这个方法。一个典型的例子就是[flash attention v2里面](https://github.com/Dao-AILab/flash-attention/blob/2dd8078adc1d9b74e315ee99718c0dea0de8eeb6/csrc/flash_attn/src/flash_fwd_kernel.h#L1183)的LSE trick（后续计算使用exp抵消log），省掉了一个对角矩阵的乘法运算。Pytorch中 [CTC loss](https://github.com/pytorch/pytorch/blob/781d28e2655f88ae2fef827ed110f22ed553a0ab/aten/src/ATen/native/cuda/LossCTC.cu#L164)、[log-sigmoid](https://github.com/alykhantejani/pytorch/blob/f7c6ba67afdd6885e1efb96540906364aa506f9c/torch/lib/THNN/generic/LogSigmoid.c#L13)里面也都用了这个trick。而且更有意思的事情是，LSE的导函数就是softmax。

到这里，我们再深入看一下交叉熵损失，[torch的doc](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)里面有提到，交叉熵损失等价于对输入做log-softmax，然后再使用nll (negative log likelihood) loss，这个实现方式比起直接套用交叉熵损失的写法，也会更加数值稳定一些。

<br>

#### softplus

softplus主要用于需要**平滑非负输出**的场景，有时候会作为relu的一个平滑替代，或者解决dying relu问题（虽然在deep learning book中[6.3.3的最后部分](https://www.deeplearningbook.org/contents/mlp.html)明确提到了虽然softplus比relu处处可导、饱和程度低，但实际上并没有比relu好），这个激活函数在[VAE中比较常见一些](https://gist.github.com/hunter-heidenreich/9512636394a23721452046039dd52d90#file-vae-py-L30)。

softplus的数学表达式为：

$$
\mathrm{softplus}(x) = \log(1 + e^x) = \mathrm{log1p}(e^x)
$$

softplus的数值不稳定性来自于exp，因为指数增长地非常快，很容易overflow成为inf，所以诸如[pytorch里面](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html)会设置一个阈值，当满足条件的时候会将softplus(x)的值设为x，比如在python中如果执行`math.log(1 + math.exp(34))`返回的结果就是34。
所以很自然地，softplus的数值稳定的写法是进行分段。

$$
\text{softplus}(x) =\begin{cases} x, & x > 20 \quad (\text{避免overflow}) \\\log(1 + e^x), & -20 \leq x \leq 20 \\e^x, & x < -20 \quad (\text{避免underflow})\end{cases}
$$

但是实际上还有另外一个写法，[megengine](https://github.com/MegEngine/MegEngine/blob/47952c075d868665e1116214bea760d786144081/imperative/python/megengine/xla/rules/elemwise.py#L404)和[FlagGems](https://github.com/FlagOpen/FlagGems/blob/93713395eb1e64f13e8ad021b2026e47d7f9d64f/src/flag_gems/ops/log_sigmoid.py#L12)里面都采用了这样的写法，表达式上也优雅了许多，不再需要手动设置阈值：

$$
\mathrm{softplus}(x) = \log(1 + e^{-|x|}) + \max(0, x) = \mathrm{log1p}(e^{-|x|}) + \max(0, x)
$$

有了数值稳定版本的softplus，进而就可以衍生出来一些使用softplus的数值稳定写法，下面我来介绍一些比较常见的，这些方法通常和sigmoid相关。

首先第一个就是log sigmoid，直接就可以用-softplus(-x)做替代。

$$
\log\text{-}\mathrm{sigmoid}(x) = \log \sigma(x) = \log \left( \frac{1}{1 + e^{-x}} \right) = -\log(1 + e^{-x}) = -\mathrm{softplus}(-x)
$$

其次是[sigmoid transform](https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/distributions/transforms.py#L609)，用来做概率分布的变换。当涉及到两个概率分布的时候，通常会牵扯到ladj（log abs det jacobian）的概念，而在[torch pr 19802](https://github.com/pytorch/pytorch/pull/19802)里面有提到过sigmoid transform的数值稳定问题。

假设 $y = \sigma(x) = \frac{1}{1 + e^{-x}}$ 表示映射关系，那么逆变换则为 $x = \mathrm{logit}(y) = \log(y) - \log(1-y)$，ladj对应的表达式为

$$
\begin{aligned}
 \log |\det J| 
&= \log(\frac{d}{dx} \sigma(x)) 
\newline
&= \log(y) + \log(1 - y) = -\log(\frac{1}{y} + \frac{1}{1-y}), \quad \text{从}y\text{的视角}
\newline
&= \log\sigma(x) + \log(1 - \sigma(x)) = -\mathrm{softplus}(-x) -\mathrm{softplus}(x), \quad \text{从}x\text{的视角}
\end{aligned}
$$

显然，softplus的写法（从x视角）要比log的写法（从y视角）更加数值稳定。因为当y接近1或者0的时候（考虑到sigmoid的饱和区间，还是很容易接近0或者1的），显然倒数的写法会更不稳定，而softplus的版本在更大的范围上性质良好。

如果仔细看softplus函数的表达式，也可以将其表示为log1pexp函数，对应的也有log1mexp函数，也就是 
$\log(1 - e^x)$ 。虽然有[issue](https://github.com/pytorch/pytorch/issues/39242)提到这个函数，但是在pytorch里面并没有这个实现，这里有一个[note](https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf)介绍log1mexp的数值稳定写法，里面也涉及到softplus的数值稳定写法，感兴趣的可以自己去看看，我们此处不再展开。

softplus是sigmoid的好伙伴，就如同LSE和softmax之间的关系一样，softplus的导函数恰好也是sigmoid。

<br>

### 后记

数值稳定性是我在过去的一段时间里比较好奇的一个问题，在之前[讨论rag的博客](https://fatescript.github.io/blog/2024/LLM-RAG/)中，我曾经抛出过类似的问题给大模型，但是一直没有得到比较满意的回答。在经过了近一年的积累和检索一些issue/pr，我自己也能获得这个问题的相对满意的答案了。

也许未来有一天，[deep research](https://openai.com/index/introducing-deep-research/)会逐渐取代我的工作方式，但我衷心希望，AI不会剥夺人类探索和创造的乐趣。这次对数值稳定性的探索，让我想起小时候拆解家里老式机械钟表的过程，也许会迷失在齿轮的迷宫里，但在弹簧突然弹开的“咔嗒”声中，我知道我搞定了一切。

**一种纯粹和天赐的快乐。**

### Reference

[1] [DeepStability: A Study of Unstable Numerical Methods and Their Solutions in Deep Learning](https://arxiv.org/pdf/2202.03493)  
[2] [stack exchange上对于lerp实现的讨论](https://math.stackexchange.com/questions/907327/accurate-floating-point-linear-interpolation/1798323#1798323)  
[3] [welford算法](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)  
[4] [C++实现的简单版本log1p](https://www.johndcook.com/blog/cpp_log_one_plus_x/)  
[5] [C++实现的简单版本expm1](https://www.johndcook.com/cpp_expm1.html)  
[6] [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)  
[7] [介绍log1mexp 数值稳定的note](https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf)  
