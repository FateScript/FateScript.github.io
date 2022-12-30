---
layout: post
title: 关于炼丹，你是否知道这些细节？
date: 2021-12-28 11:59:00-0400
description: 
categories: deep-learning computer-vision
tags: [code,]
giscus_comments: true
---

本文算是我工作一年多以来的一些想法和经验，最早发布在旷视研究院内部的论坛中，本着开放和分享的精神发布在我的知乎专栏中，如果想看干货的话可以直接跳过动机部分。另外，后续在这个专栏中，我会做一些关于原理和设计方面的一些分享，希望能给领域从业人员提供一些看待问题的不一样的视角。


## 动机
前段时间走在路上，一直在思考一个问题：我的时间开销很多都被拿去给别人解释一些在我看起来显而易见的问题了，比如[cvpods](https://github.com/Megvii-BaseDetection/cvpods)里面的一些code写法问题（虽然这在某些方面说明了文档建设的不完善），而这变相导致了我实际工作时间的减少，如何让别人少问一些我觉得答案显而易见的问题？如何让别人提前规避一些不必要的坑？只有解决掉这样的一些问题，我才能从一件件繁琐的小事中解放出来，把精力放在我真正关心的事情上去。

其实之前同事有跟我说过类似的话，每次带一个新人，都要告诉他：你的实现需要注意这里blabla，还要注意那里blabla。说实话，我很佩服剑锋同学带intern的细致和知无不言，但我本性上并不喜欢每次花费时间去解释一些我觉得显而易见的问题，所以我打算写一个帖子，把我踩过的坑和留下来的经验broadcast出去。希望能够方便别人，同时也节约我的时间。

加入旷视以来，个人一直在做一些关于框架相关的内容，所以内容主要偏向于模型训练之类的工作。因为**我无法想象知识在别人脑海中的样子（the curse of knowledge），所以只能选取被问的最多的，和我觉得最应该知道的**。

准备好了的话，我们就启航出发（另，这篇blog会长期进行更新）。


## 坑/经验

### Data模块
1. python图像处理用的最多的两个库是opencv和Pillow（PIL），但是两者读取出来的图像并不一样，**opencv读取的图像格式的三个通道是BGR形式的，但是PIL是RGB格式的**。这个问题看起来很小，但是衍生出来的坑可以有很多，最常见的场景就是数据增强和预训练模型中。比如有些数据增强的方法是基于channel维度的，比如megengine里面的[HueTransform](https://github.com/MegEngine/MegEngine/blob/4d72e7071d6b8f8240edc56c6853384850b7407f/imperative/python/megengine/data/transform/vision/transform.py#L937)，在[这一行](https://github.com/MegEngine/MegEngine/blob/4d72e7071d6b8f8240edc56c6853384850b7407f/imperative/python/megengine/data/transform/vision/transform.py#L958)显然是需要确保图像是BGR的，但是经常会有人只看有Transform就无脑用了，从来没有考虑过这些问题。
2. 接上条，RGB和BGR的另一个问题就是导致预训练模型载入后训练的方式不对，最常见的场景就是预训练模型的input channel是RGB的（例如torch官方来的预训练模型），然后你用cv2做数据处理，最后还忘了convert成RGB的格式，那么就是会有问题。这个问题应该很多炼丹的同学没有注意过，我之前写[CenterNet-better](https://github.com/FateScript/CenterNet-better)就发现[CenterNet](https://github.com/xingyizhou/CenterNet)存在这么一个问题，要知道当时这可是一个有着3k多star的仓库，但是从来没有人意识到有这个问题。当然，依照我的经验，如果你训练的iter足够多，即使你的channel有问题，对于结果的影响也会非常小。不过，既然能做对，为啥不注意这些问题一次性做对呢？
3. torchvision中提供的模型，都是输入图像经过了ToTensor操作train出来的。也就是说最后在进入网络之前会统一除以255从而将网络的输入变到0到1之间。torchvision的[文档](https://pytorch.org/vision/stable/models.html)给出了他们使用的mean和std，也是0-1的mean和std。如果你使用torch预训练的模型，但是输入还是0-255的，那么恭喜你，在载入模型上你又会踩一个大坑（要么你的图像先除以255，要么mean和std都要乘以255）。
4. ToTensor之后接数据处理的坑。上一条说了ToTensor之后图像变成了0到1的，但是一些数据增强对数值做处理的时候，是针对标准图像，很多人ToTensor之后接了这样一个数据增强，最后就是练出来的丹是废的（心疼电费QaQ）。
5. 数据集里面有一个图特别诡异，只要train到那一张图就会炸显存（CUDA OOM），别的图训练起来都没有问题，应该怎么处理？通常出现这个问题，首先判断数据本身是不是有问题。如果数据本身有问题，在一开始生成Dataset对象的时候去掉就行了。如果数据本身没有问题，只不过因为一些特殊原因导致显存炸了（比如检测中图像的GT boxes过多的问题），可以catch一个CUDA OOM的error之后将一些逻辑放在CPU上，最后retry一下，这样只是会慢一个iter，但是训练过程还是可以完整走完的。
6. pytorch中dataloader的坑。有时候会遇到pytorch num_workers=0（也就是单进程）没有问题，但是多进程就会报一些看不懂的错的现象，这种情况通常是因为torch到了ulimit的上限，更核心的原因是**torch的dataloader不会释放文件描述符**（参考[issue](https://github.com/pytorch/pytorch/issues/973)）。可以ulimit -n 看一下机器的设置。跑程序之前修改一下对应的数值。
7. opencv和dataloader的神奇联动。很多人经常来问为啥要写cv2.setNumThreads(0)，其实是因为cv2在做resize等op的时候会用多线程，当torch的dataloader是多进程的时候，多进程套多线程，很容易就卡死了（具体哪里死锁了我没探究很深）。除了setNumThreads之外，通常还要加一句cv2.ocl.setUseOpenCL(False)，原因是cv2使用opencl和cuda一起用的时候通常会拖慢速度，加了万事大吉，说不定还能加速。
8. dataloader会在epoch结束之后进行类似重新加载的操作，复现这个问题的code放在后面的 code复现部分了。这个问题算是可以说是一个高级bug/feature了，可能导致的问题之一就是炼丹师在本地的code上进行了一些修改，然后训练过程直接加载进去了。解决方法也很简单，让你的sampler源源不断地产生数据就好，这样即使本地code有修改也不会加载进去。


### Module模块
1. BatchNorm在训练和推断的时候的行为是不一致的。这也是新人最常见的错误（类似的算子还有dropout，这里提一嘴，**pytorch的dropout在eval的时候行为是Identity**，之前有遇到过实习生说dropout加了没效果，直到我看了他的code： x = F.dropout(x, p=0.5)  ）

2. BatchNorm叠加分布式训练的坑。**在使用DDP（DistributedDataParallel）进行训练的时候，每张卡上的BN统计量是可能不一样的，仔细检查broadcast_buffer这个参数**。DDP的默认行为是在forward之前将rank0 的 buffer做一次broadcast（broadcast_buffer=True），但是一些常用的开源检测仓库是将broadcast_buffer设置成False的（参考：[mmdet](https://github.com/facebookresearch/detectron2/blob/f50ec07cf220982e2c4861c5a9a17c4864ab5bfd/tools/plain_train_net.py#L206) 和 [detectron2](https://github.com/facebookresearch/detectron2/blob/f50ec07cf220982e2c4861c5a9a17c4864ab5bfd/tools/plain_train_net.py#L206)，我猜是在检测任务中因为batchsize过小，统一用卡0的统计量会掉点）**这个问题在一边训练一边测试的code中更常见**，比如说你train了5个epoch，然后要分布式测试一下。一般的逻辑是将数据集分到每块卡上，每块卡进行inference，最后gather到卡0上进行测点。但是**因为每张卡统计量是不一样的，所以和那种把卡0的模型broadcast到不同卡上测试出来的结果是不一样的。这也是为啥通常训练完测的点和单独起了一个测试脚本跑出来的点不一样的原因**（当然你用SyncBN就不会有这个问题）。
3. Pytorch的SyncBN在1.5之前一直实现的有bug，所以存在使用SyncBN结果掉点的问题。
4. 用了多卡开多尺度训练，明明尺度更小了，但是速度好像不是很理想？这个问题涉及到多卡的原理，因为分布式训练的时候，在得到新的参数之后往往需要进行一次同步。假设有两张卡，卡0的尺度非常小，卡1的尺度非常大，那么就会出现卡0始终在等卡1，于是就出现了虽然有的尺度变小了，但是整体的训练速度并没有变快的现象（木桶效应）。解决这个问题的思路就是**尽量把负载拉均衡一些**。
5. 多卡的小batch模拟大batch（梯度累积）的坑。假设我们在单卡下只能塞下batchsize = 2，那么为了模拟一个batchsize = 8的效果，通常的做法是forward / backward 4次，不清理梯度，step一次（当然考虑BN的统计量问题这种做法和单纯的batchsize=8肯定还是有一些差别的）。在多卡下，因为调用loss.backward的时候会做grad的同步，所以说前三次调用backward的时候需要加ddp.no_sync的context manager（不加的话，第一次bp之后，各个卡上的grad此时会进行同步），最后一次则不需要加。当然，我看很多仓库并没有这么做，我只能理解他们就是单纯想做梯度累积（BTW，加了[ddp.no_sync](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html?highlight=no_sync#torch.nn.parallel.DistributedDataParallel.no_sync)会使得程序快一些，毕竟加了之后bp过程是无通讯的）。
6. **浮点数的加法其实不遵守交换律的**，这个通常能衍生出来GPU上的运算结果不能严格复现的现象。可能一些非计算机软件专业的同学并不理解这一件事情，直接自己开一个python终端体验可能会更好：
{% highlight python %}
print(1e100 + 1e-4 + -1e100)  # ouptut: 0
print(1e100 + -1e100 + 1e-4)  # output: 0.0001
{% endhighlight %}


## 训练模块
1. FP16训练/混合精度训练。使用Apex训练混合精度模型，在保存checkpoint用于继续训练的时候，除了model和optimizer本身的state_dict之外，还需要保存一下amp的state_dict，这个在[amp的文档](https://nvidia.github.io/apex/amp.html#checkpointing)中也有提过。（当然，经验上来说忘了保存影响不大，会多花几个iter search一个loss scalar出来）
2. 多机分布式训练卡死。 @zhangsongyang  遇到的一个坑。场景是rlaunch申请了两个8卡机，然后机器1和机器2用前4块卡做通讯（local rank最大都是4）。可以初始化process group，在使用DDP的时候会卡死。原因在于pytorch在做DDP的时候会猜测一个rank，参考[code](https://github.com/pytorch/pytorch/blob/0d437fe6d0ef17648072eb586484a4a5a080b094/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1622-L1630)。对于上面的场景，第二个机器上因为存在卡5到卡8，而对应的rank也是5到8，所以DDP就会认为自己需要同步的是卡5到卡8，于是就卡死了。
3. 在使用AMP的时候，使用Adam/AdamW优化器之后NaN，之前没有任何异常现象，通常是optimizer里面的eps的问题，调整一下eps的数值就好了（比如1e-3），因为默认的eps是1e-8，在fp16下浮点运算容易出NaN
4. **梯度为0** 和 **参数是否更新** 没有必然关系。因为grad并不是最终的参数更新量，最终的参数更新量是在optimizer里面进行计算的。一个最简单的例子就是设置了weight decay不为0，当optimizer的weight decay不为0 的时候，最终的参数更新量都会加上 `lr * wd * param` ，所以 grad为0并不等价于参数量不会更新。一些可以refer的[code](https://github.com/MegEngine/MegEngine/blob/d404ed184d/imperative/python/megengine/optimizer/sgd.py#L72-L73)（此处以megengine为例，pytorch仅仅是把逻辑写成了cpp来加速）


## 复现Code

### Data部分
{% highlight python linenos %}
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tqdm
import time


class SimpleDataset(Dataset):
    def __init__(self, length=400):
        self.length = length
        self.data_list = list(range(length))

    def __getitem__(self, index):
        data = self.data_list[index]
        time.sleep(0.1)
        return data

    def __len__(self):
        return self.length


def train(local_rank):
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)
    iter_loader = iter(dataloader)
    max_iter = 100000
    for _ in tqdm.tqdm(range(max_iter)):
        try:
            _ = next(iter_loader)
        except StopIteration:
            print("Refresh here !!!!!!!!")
            iter_loader = iter(dataloader)
            _ = next(iter_loader)
            

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.spawn(train, args=(), nprocs=2, daemon=False)
{% endhighlight %}

当程序运行起来的时候，可以在Dataset里面的\_\_getitem\_\_里面加一个print，在refresh之后，就会print内容（看到现象是不是觉得自己以前炼的丹可能有问题了呢）。

## 碎碎念

一口气写了这么多条也有点累了，后续有踩到新坑的话我也会继续更新这篇文章的。毕竟写这篇文章是希望工作中不再会有人踩类似的坑 & 炼丹的人能够对深度学习框架有意识（虽然某种程度上来讲这算是个心智负担）。

如果说今年来什么事情是最大的收获的话，那就是理解了一个开放的生态是可以迸发出极强的活力的，也希望能看到更多的人来分享自己遇到的问题和解决的思路。毕竟探索的答案只是一个副产品，过程本身才是最大的财宝。