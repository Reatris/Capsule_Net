# 背景简介

Geoffrey Hinton，深度学习的开创者之一，反向传播等神经网络经典算法的发明人，2017年10月发表了论文，介绍了全新的胶囊网络模型，以及相应的囊间动态路由算法。

论文[https://arxiv.org/pdf/1710.09829.pdf](https://arxiv.org/pdf/1710.09829.pdf)
            
 https://ai-studio-static-online.cdn.bcebos.com/7a885d651bd6415bbe6826015723bc14ab1d4c1e9dc542579c2ff03294951a04


Geoffrey Hinton的胶囊网络（Capsule Network）一经发布就震动了整个人工智能领域，它将卷积神经网络（CNN）的极限提升到一个新的水平。这种网络基于一种被Hinton称为胶囊（capsule）的结构。 此外，他还发表了囊间动态路由算法，用来训练新提出的胶囊网络。让我们一起来看看他的这种网络结构和原理。
Capsule网络结构，不仅可以和卷积神经网络一样用来处理视觉问题，同样也可以应用到其他领域。首先让我们先来关注Capsule网络是如何克服CNN存在的问题的。
用Hinton大佬的话来说，计算机图形学做的是渲染，而计算视觉想做的就是渲染的逆过程。渲染是将三维的图像投影到二维，在数学上意味着给原图乘以一个固定的矩阵。而计算机视觉做的，则是Inverse Graphics，也就是从二维的图像推测出本身的三维结构（输入多个不同位姿的二维图像，就能形成三维对象，是不是和人很接近呢）。为了比较胶囊网络和CNN之间的区别，让我们先看看CNN想做什么，有什么不足。


# 卷积神经网络的不足之处

* CNN（卷积神经网络）的表现是如此优异，以至于深度学习现在如此流行。但是把检测目标的平移，旋转，加上边框等干扰会被CNN识别成其他目标，列如CNN会认为下图的三个R是不同的字母，如果使用暴力方法，把各个角度的样本都囊括进去，这样使得CNN所需的训练集要变得很大，而数据增强技术虽然有用，但提升有限，无法从根本上解决问题。

 https://ai-studio-static-online.cdn.bcebos.com/ef4e879245f14d33ae1b606cc0bcb04170fe0efbd2054d20a05abc1dae7ebd63


让我们再考虑一个非常简单的例子。如下图，如果有一张脸，那么它是由哪些特征构成的？椭圆的轮廓、眼睛、鼻子和一个嘴巴。CNN可以轻而易举地检测到这些特征，并且因此认为它检测到的是脸。但是当你用CNN去识别右边这张脸(眼睛和嘴巴位置改变了)依然会得到同样的结果。这是因为CNN识别脸时，仅仅只识别脸的几个特征部分，右图中的确有两个眼睛，一个鼻子，一张嘴，虽然位置不对，但是CNN一旦检测到这些特征，那么识别结果是就是脸， CNN是不会注意子结构之间关系的。

 https://ai-studio-static-online.cdn.bcebos.com/968b9c6420834a2392e89f4c6fee3a563f7a47e5c41942979eb9171acffea59b


究其原因是CNN的主要部分是卷积层，用于检测图像像素中的重要特征。较深的层（更接近输入的层）将学习检测诸如边缘和颜色渐变之类的简单特征，而较高的层则将简单特征组合成复杂一些的特征。最后，网络顶部的致密层组合高层特征并输出分类预测。

但凡对CNN有所了解都知道，低层特征通过加权组合成高层特征。不过在这个过程中，组成高层特征的低层特征之间并不存在位姿（平移和旋转）关系。为了解决这个问题，CNN通过后接最大池化层或者后续卷积层。这样不仅能减少参数，还能增加网络神经元的视野，检测更大区域的特征来弥补，如此达到的效果在某些领域甚至能超越人类。但是Hinton自己就表示，卷积神经网络使用的池化操作是一个巨大的错误，它表现地如此优异则是一场灾难。不仅如此还有一个关键问题：卷积神经网络的内部数据表现出它不能形成简单和复杂对象之间的重要空间层级。就如上面的例子，图片中存在两只眼睛、一张嘴和一个鼻子，仅仅这些并不意味着图片中存在一张脸，还需要考虑这些对象彼此之间的朝向关系。

不仅如此，Hinton认为的人与CNN神经网络的最大区别：人类在识别图像的时候，是遵照树形的方式，由上而下展开式的，而CNN则是通过一层层的过滤，将信息一步步由下而上的进行抽象。



 https://ai-studio-static-online.cdn.bcebos.com/1b02268e441d4d969c9dfaf16dc3121f7834eb75e61f4bfca5ac2120399909e2

正因为卷积神经网络神经元之间都是平等的，缺少一种内部结构，所以对不同位置、角度下的同一个物品可能做出不同识别，更无法表现子结构之间的关系。CNN中采用的分块和共享权重的方法，以使其够使神经网络学到的特征提取能够在图形出现微小变化时能够应对，而不是针对图形的变化，对应神经网络进行相应的改变，而这正是capsule神经网络所要做的。



# 位姿

Hinton主张，为了正确地分类和辨识对象，保留对象部件间的分层位姿关系很重要。这是让你理解胶囊理论为何如此重要的关键所在，它结合了对象之间的相对关系，在数值上表示为4维位姿矩阵。所以我们要在神经网络中尝试建立位姿关系，在三维图形中，三维对象之间的关系可以用位姿表示，位姿的本质是平移和旋转
当在神经网络里面构建了这些关系之后，模型就能非常容易理解他看到的是以前的东西，只不过是另一个视角而已。从下面的图片中你可以轻易辨识出这是自由女神像，尽管所有的图像显示的角度都不一样。这是因为你脑中的自由女神像的内部表示并不依赖视角。你大概从没有见过和这些一模一样的图片，但你仍然能立刻知道这是自由女神像。


 https://ai-studio-static-online.cdn.bcebos.com/037eb95ea3dc4e53837579cd3b03756c0c9c1f5e8dd6418da085f14b23b1720d

但是对CNN而言，这个任务非常难，因为它没有内建对三维空间的理解。而对于胶囊神经网络而言，这个任务要容易得多，因为它显式地建模了这些关系。相比之前最先进的方法，使用CapsNet的论文能够将错误率降低，这是一个巨大的提升。胶囊方法的另一大益处在于，相比CNN需要的数据，它只需要学习一小部分数据，就能达到最先进的效果（Hinton在他关于CNN错误的著名演说中提到了这一点）。 从这个意义上说，胶囊理论实际上更接近人脑的行为。为了学会区分数字，人脑只需要几十个例子，最多几百个例子。而CNN则需要几万个例子才能取得很好的效果。这看起来像是在暴力破解，显然要比我们的大脑低级。

下图为胶囊神经网络的位姿辨别效果

 https://ai-studio-static-online.cdn.bcebos.com/6899882519aa46b284908b6764a4fecd44f6f09f30964e8d962daab8d9936396

和其他模型相比，胶囊网络在辨识上一列和下一列的图片属于同一类、仅仅视角不同方面，表现要好很多，相对于CNN这是压倒性的优势


# 胶囊是什么？

让我们先看看Hinton等人的《Transforming Autoencoders》中关于胶囊的描述：

`
人工神经网络不应当追求“神经元”活动中的视角不变性（使用单一的标量输出来总结一个局部池中的重复特征检测器的活动），而应当使用局部的“胶囊”，这些胶囊对其输入执行一些相当复杂的内部计算，然后将这些计算的结果封装成一个包含信息丰富的输出的小向量。每个胶囊学习辨识一个**有限的观察条件和变形范围内隐式定义的视觉实体**，并输出实体在有限范围内存在的概率及一组“实例参数”，实例参数可能包括相对这个视觉实体的隐式定义的典型版本的精确的位姿、照明条件和变形信息。当胶囊工作正常时，视觉实体存在的概率具有局部不变性——当实体在胶囊覆盖的有限范围内的外观流形上移动时，概率不会改变。实例参数却是“等变的”——随着观察条件的变化，实体在外观流形上移动时，实例参数也会相应地变化，因为实例参数表示实体在外观流形上的内在坐标。
`

阅读上面这段话我们可以很好理解到，人造神经元输出单个标量表示结果，而胶囊可以输出向量作为结果，CNN（卷积神经网络）使用卷积层获取特征矩阵，为了在神经元的活动中实现视角不变性。我们通过最大池化方法来达成这一点。最大池化持续地搜寻二维特征矩阵的区域，以选取每个区域中最大的数字作为输出结果。如果我们略微调整输入，在输入图像上，我们稍微变换一下我们想要检测的对象时，由于最大池化保持不变，网络仍然能检测到对象。

使用最大池的缺点就是丢失了有价值的信息，也没能处理特征之间的相对空间关系。但是胶囊检测中的特征的状态的重要信息，都将以向量的形式被胶囊封装。

胶囊将特征检测的概率作为其输出向量的长度进行编码。检测出的特征的状态被编码为该向量指向的方向。所以，当检测出的特征在图像中移动或其状态不知怎的发生变化时，概率仍然保持不变（向量长度没有改变），但它的方向改变了。想象一个胶囊，它检测图像中的面部，并输出长度小于1的三维向量。接着我们开始在图像上移动面部。向量将在空间上旋转，表示检测出的面部的状态改变了，但其长度（检测概率）仍然保持，这使胶囊仍然确信它检测出到了面部。这就是Hinton所说的活动等变性：这才是我们应该追求的那种不变性，而不是CNN提供的基于最大池化的不变性。


# 胶囊的工作原理

让我们比较下胶囊与人造神经元。下表中Vector表示向量，scalar表示标量，Operation中对比了它们工作原理的差异。

 https://ai-studio-static-online.cdn.bcebos.com/854f186b6446428ab2b6784e4bdfcd11f7d7d70922bc49ea8d6a4327cd282a9b

图中Vector表示向量，scalar表示标量，Operation中对比了它们的不同工作原理

人造神经元可以用3个步骤来表示：

1. 输入标量的标量加权

2. 加权输入标量之和

3. 标量到标量的非线性变换

胶囊具有上面3个步骤的向量版，并新增了输入的仿射变换这一步骤：

1.输入向量的矩阵乘法

2.输入向量的标量加权

3.加权输入向量之和

4.向量到向量的非线性变换


**下面将详细说明这4个步骤**


# 1.输入向量的矩阵乘法


 https://ai-studio-static-online.cdn.bcebos.com/5a43f7e99cb444ca9e3e6a303da9bba9fefd2e32c180400eac70d97882bdec67


胶囊接收的输入向量（上图中的U1、U2和U3）来自下层的3个胶囊。这些向量的长度分别编码下层胶囊检测出的相应特征的概率，向量的方向则编码检测出的特征的一些内部状态。让我们假定下层的胶囊分别检测眼睛、嘴巴和鼻子，而输出胶囊检测面部。接着将这些向量乘以相应的权重矩阵W，W编码了低层特征（眼睛、嘴巴和鼻子）和高层特征（面部）之间的空间关系和其他重要关系。乘以这些矩阵后，我们得到的是高层特征的状态（位置，方向，大小等），你也可以理解为，û1表示根据检测出的眼睛的位置，面部应该在什么位置，û2表示根据检测出的嘴巴的位置，面部应该在什么位置，û3表示根据检测出的鼻子的位置，面部应该在什么位置。如果这3个胶囊输出对象（面部）位置相同，那么就可以将这3个输出编码出一个更高层的特征(同时关于眼睛、嘴巴、鼻子、面部的关系特征）


# 2.输入向量的标量加权

* **一个底层胶囊搞如何把信息输出给高级胶囊呢**

之前的人造神经元是通过反向传播算法一步步调整权重优化网络，而胶囊则有所不同

 https://ai-studio-static-online.cdn.bcebos.com/46700a79bd1349f9b66be71a9c872c31e983f1ad91e34f34be50fafc0c0c867a


上图中，左右分别是高层的两个不同胶囊，方形区域内的点则是下层胶囊输入在这个胶囊的分布，一个低层胶囊需要“决定”将它的输出发送给哪个高层胶囊。它将通过调整权重C做出决定，胶囊在发送输出前，先将输出乘以这个权重。胶囊将决定是把输出发给左边的胶囊J，还是发给右边的胶囊K。

关于权重，我们需要知道：

1. 权重均为非负标量（因为经过softmax函数加权）。

2. 对每个低层胶囊i而言，所有权重的总和等于1（因为经过softmax函数加权）。

3. 对每个低层胶囊i而言，权重的数量等于高层胶囊的数量。

4. 这些权重的数值由迭代动态路由算法确定。


对于每个低层胶囊i而言，其权重定义了传给每个高层胶囊j的输出的概率分布。许多个低层胶囊通过加权把向量输入高层胶囊，同时高层胶囊就会接收到来自其他低层胶囊的许多向量。所有这些输入以红点和蓝点表示。这些点聚集的地方，意味着低层胶囊的预测互相接近。比如，胶囊J和K中都有一组聚集的红点，因为那些胶囊的预测很接近。所以，一个低层胶囊该把它的输出发给胶囊J还是胶囊K呢？这个问题的答案正是动态路由算法的精髓。低层胶囊的输出乘以相应的矩阵W后，落在了远离胶囊J中的红色聚集区的地方，另一方面，在胶囊K中，它落在红色聚集区边缘，红色聚集区表示了这个高层胶囊的预测结果。低层胶囊具备测量哪个高层胶囊更能接受其输出的机制，并据此自动调整权重，使对应胶囊K的权重C变高，对应胶囊J的权重C变低。



# 3.加权输入向量之和
这一步骤表示输入的组合，和通常的人工神经网络差不多，除了它是向量的和而不是标量的和。



# 4. 向量到向量的非线性变换

CapsNet的另一大创新是新颖的非线性激活函数，这个函数接受一个向量，然后在不改变方向的前提下，压缩它的长度到1以下。

 https://ai-studio-static-online.cdn.bcebos.com/ce20ab13bad746ccbc5aef7033328a0cade49546e16c4d27bb4b42b9945a371b

*  ▲||Sj||表示模长

* **上面这个公式：向量经过转换之后小于1个单位向量**

* **方向不变(对单位向量长度缩放）**

* **原来向量模越大，经过激活函数后的模长越接近1**

```
def squash(self,vector):
     	 '''
        压缩向量的函数，类似激活函数，向量归一化
        Args:
            vector：一个4维张量 [batch_size,vector_num,vector_units_num,1]
        Returns:
            一个和x形状相同，长度经过压缩的向量
            输入向量|v|（向量长度）越大，输出|v|越接近1
        '''
        vec_abs = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(vector)))  # 一个标量|v|模长度
        scalar_factor = fluid.layers.square(vec_abs) / (1 + fluid.layers.square(vec_abs))  #公式中左边做商部分
        vec_squashed = scalar_factor * fluid.layers.elementwise_div(vector, vec_abs)    # 对应元素相除
        return(vec_squashed)
```

# 囊间动态路由算法(精髓所在）

* 低层胶囊将其输出发送给对此表示“同意”的高层胶囊。这是动态路由算法的精髓。

 https://ai-studio-static-online.cdn.bcebos.com/06275f20518648f883611e087ef7889117af8655dc784e1ebd33b8e50077d901


**▲囊间动态路由算法伪代码**

* 伪代码的第一行指明了算法的输入：低层输入向量经过矩阵乘法得到的û，以及路由迭代次数r。最后一行指明了算法的输出，高层胶囊的向量vj。

* 第2行的bij是一个临时变量，存放了低层向量对高层胶囊的权重，它的值会在迭代过程中逐个更新，当开始一轮迭代时，它的值经过softmax转换成cij。在囊间动态路由算法开始时，bij的值被初始化为零(但是经过softmax后会转换成非零且各个权重相等的cij)。

```
#路由分配权重 b_ij的初始化
#u_hat_num:低层向量数目
#cap_num:高层胶囊数目
B_ij = fluid.layers.ones((1,u_hat_num,cap_num,1),dtype='float32')/cap_num
```

* 第3行表明第4-7行的步骤会被重复r次（路由迭代次数）。

* 第4行计算低层胶囊向量i的对应所有高层胶囊的权重。bi的值经过softmax后会转换成非零权重ci且其元素总和等于1。


```
#softmax过程
C_ij = fluid.layers.softmax(B_ij,axis=2)
```

* 如果是第一次迭代，所有系数cij的值会相等。例如，如果我们有8个低层胶囊和10个高层胶囊，那么所有cij的权重都将等于0.1。这样初始化使不确定性达到最大值：低层胶囊不知道它们的输出最适合哪个高层胶囊。当然，随着这一进程的重复，这些均匀分布将发生改变。

* 第5行，那里将涉及高层胶囊。这一步计算经前一步确定的路由系数ci加权后的输入向量的总和，得到输出向量sj。

```
#使用元素逐一乘算加权，比如(1,1152,1,1)*(32,1152,16,1)-->（32,1152，16,1）每16个向量分配一个权重
v_j = fluid.layers.elementwise_mul(u_hat,c_ij)
#将分配到这一个胶囊的向量相加得到v_j:(32,1,16,1)的输出
v_j = fluid.layers.reduce_sum(v_j,dim=1,keep_dim=True)
```

* 第6行，来自前一步的向量将穿过squash非线性函数，反向不变，长度被归一化至1以下。

```
v_j = self.squash(v_j) 
```

* 第7行进行更新权重，这是路由算法的精髓所在。我们将每个高层胶囊的向量vj与低层原来的输入向量û逐元素相乘求和获得内积（也叫点积，点积检测胶囊的输入和输出之间的相似性（下图为示意图）），再用点积结果更新原来的权重bi。这就达到了’低层胶囊将其输出发送给具有类似输出的高层胶囊’的效果，点积刻画了向量之间的相似性。这一步骤之后，算法跳转到第3步重新开始这一流程，并重复r次。

```
#平铺v_j (32,1,16,1)-->(32,1152,16,1) 因为这要对1152个不同向量进行计算
v_j_expand = fluid.layers.expand(v_j,(1,pre_cap_num,1,1))
#求内积 也是逐一元素相乘算，然后求和 #(32,1152,16,1)-->(32,1152,1,1)
u_v_produce = fluid.layers.elementwise_mul(u_hat,v_j_expand)
u_v_produce = fluid.layers.reduce_sum(u_v_produce,dim=2,keep_dim=True)
#内积累加(把bach_size的累加到一块)，更新路由权重BIJ
b_ij += fluid.layers.reduce_sum(u_v_produce,dim=0,keep_dim=True)
```


 https://ai-studio-static-online.cdn.bcebos.com/5bffe77ab8cc41a28168b7fc76ad7f82334452a35c3647338494ff5f3510a850


**▲点积运算即为向量的内积（点积）运算，可以表现向量的相似性,点积运算接收两个向量，并输出一个标量。对于给定长度但方向不同的两个向量而言，点积有几种情况：
a正值（夹角小于90°）；b零（夹角垂直）；c负值（夹角大于180°）**

 https://ai-studio-static-online.cdn.bcebos.com/b0ae75024fae430798196c7b4c9deb9a1c1fd6f0bd4e4e12b97073b5784ecd5c

上图中，两个高层胶囊的输出用向量v1和v2表示。橙色向量表示接收自某个低层胶囊的输入，其他黑色向量表示接收自其他低层胶囊的输入。

我们看到，左边的输出v1和橙色输入û1|1指向相反的方向，它们并不相似。**这意味着它们的点积将是一个负数，与bi相加后值变小，并减少路由系数c11(见伪代码第7层）**。右边的输出v2和橙色输入û2|1指向相同的方向，它们是相似的，所以，路由系数c12会增加。经过路由迭代计算，得到一个路由系数的集合，使来自低层胶囊的输出与高层胶囊的输出的最佳匹配。


重复r次后，我们计算出了所有高层胶囊的输出，并确立正确路由权重。



# 损失函数

训练时，对于每个训练样本，根据下面的公式计算每个胶囊向量的损失值，然后将10个损失值相加得到最终损失。这是一个监督学习，所以每个训练样本都有正确的标签，在这种情况下，它将是一个10维one-hot编码向量，该向量由9个零和1个一（正确标签）组成。在损失函数公式中，与正确的标签对应的输出胶囊，系数Tc为1

 https://ai-studio-static-online.cdn.bcebos.com/f050bc29f2f543859a3150df60c5360577a75f87eb4446bc8a2186d14941d06f

如果正确标签是9，这意味着第9个胶囊输出的损失函数的Tc为1，其余9个为0。

当Tc为1时，公式中损失函数的右项系数为零，也就是说正确输出项损失函数的值只包含了左项计算。相应的左系数为0则右项系数为1，错误输出项损失函数的值只包含了右项计算。

|v|为胶囊输出向量的模长，一定程度上表示了类概率的大小，我们再拟定一个量m+，用这个变量来衡量概率是否合适，将m+与|v|作差，即得到了左项中的公式，正确输出项的概率（|v|）大于这个值则loss为0，越接近则loss越小。

同样的，将m-与|v|作差，即得到了右项中的公式错误输出项的概率，（|v|）小于这个值则loss为0，越接近则loss越小，公式右项包括了一个lambda系数以确保训练中的数值稳定性（lambda为固定值0.5），这两项取平方是为了让损失函数符合L2正则。

```
    def get_loss_v(self,label):
        '''
        计算边缘损失
        Args:
            label:shape=(32,10) one-hot形式的标签
        Notes:
            这里我调用Relu把小于0的值筛除
            m_plus：正确输出项的概率（|v|）大于这个值则loss为0，越接近则loss越小
            m_det：错误输出项的概率（|v|）小于这个值则loss为0，越接近则loss越小
            （|v|即胶囊(向量)的模长）
        '''
        #计算左项，虽然m+是单个值，但是可以通过广播的形式与label（32,10)作差
        max_l =  fluid.layers.relu(train_params['m_plus'] - self.output_caps_v_lenth)
        #平方运算后reshape
        max_l = fluid.layers.reshape(fluid.layers.square(max_l),(train_params['batch_size'],-1))#32,10
        #同样方法计算右项
        max_r =  fluid.layers.relu(self.output_caps_v_lenth - train_params['m_det'])
        max_r = fluid.layers.reshape(fluid.layers.square(max_r),(train_params['batch_size'],-1))#32,10
        #合并的时候直接用one-hot形式的标签逐元素乘算便可
        margin_loss = fluid.layers.elementwise_mul(label,max_l)\
                        + fluid.layers.elementwise_mul(1-label,max_r)*train_params['lambda_val']
        self.margin_loss = fluid.layers.reduce_mean(margin_loss,dim=1)

```


# 编码器

完整的网络结构分为编码器和解码器，我们先来看看编码器

 https://ai-studio-static-online.cdn.bcebos.com/676f8fe58d314c279caae90caa68fc328c89e47f390a4060bc8a5c9b19f83d1a


1.输入图片28x28首先经过1x256x9x9的卷积层 获得256个20x20的特征图

2.然后再用8组256x32x9x9(stride=2)的卷积获得8组32x6x6的特征图

3.之后将获取的特征图向量化输入10个胶囊，这10个胶囊输出向量的长度就是各个类别的概率。

```

class Capconv_Net(fluid.dygraph.Layer):
    def __init__(self):
        super(Capconv_Net,self).__init__()
        #第一个1x256x9x9的卷积层
        self.add_sublayer('conv0',fluid.dygraph.Conv2D(\
        num_channels=1,num_filters=256,filter_size=(9,9),padding=0,stride = 1,act='relu'))
        #8组256x32x9x9(stride=2)的卷积层（这里使用了for循环）
        for i in range(8):
            self.add_sublayer('conv_vector_'+str(i),fluid.dygraph.Conv2D(\
            num_channels=256,num_filters=32,filter_size=(9,9),stride=2,padding=0,act='relu'))
    
    def forward(self,x,v_units_num):
        x = getattr(self,'conv0')(x)
        capsules = []#存放胶囊向量
        for i in range(v_units_num):
            temp_x = getattr(self,'conv_vector_'+str(i))(x)
            capsules.append(fluid.layers.reshape(temp_x,(train_params['batch_size'],-1,1,1)))
        x = fluid.layers.concat(capsules,axis=2)#这拼接生成了1152个8维向量
        x = self.squash(x)
        return x

```

从实现代码中我们不难看出特征图转换成向量实际的过程是将每组二维矩阵展开成一维矩阵（当然有多个二维矩阵则展开后前后拼接）
之后再将所有组的一维矩阵在新的维度拼接形成向量（下图为示意图）

 https://ai-studio-static-online.cdn.bcebos.com/24199562bf27462d97975f34b85639cc6044139c82d54966989e2a59360a7057


当然向量化的方法我认为可以有所改进：

1.这8个32x9x9的卷积组如果换成1个256x9x9的卷积不是一样吗，为何要分开来？

根据这个疑问我把8个卷积组合成一个卷积层，后面直接reshape成向量

结果发现网络就这么失效了。然后我转变思路不用reshape，而是用split把特征图分组之后用for循环拼接，网络又有效了，也加快了速度。

2.虽然已经把8次卷积缩小到了一次卷积，但是仍然使用的for循环，在循环次数过多的情况下运行效率会变慢。

经过思索发现只用split和concat方法也可以直接向量化。

下面是我的思路：



```python
#分组卷积向量化改进(思路)
#本质上是为了跳出多重循环和卷积
a = np.reshape(np.arange(128),(2,4,4,4))
a = fluid.dygraph.to_variable(a)
print('输入的1,C,H,W')
print(a.numpy())

# d = fluid.layers.reshape(a,(1,32,2,1))
# print('直接reshape获得的向量无用')
# print(d.numpy())

b = fluid.layers.split(a,2,dim=1)
concats = []
for i in range(2):
    b[i] = fluid.layers.reshape(b[i],(2,32,-1,1))
    concats.append(b[i])
c = fluid.layers.concat(concats,axis=2)
print('向量化拼接法1：直接在后面循环拼接')
print(c.numpy())

e = fluid.layers.reshape(a,(2,2,-1))
e = fluid.layers.split(e,32,dim=2)
e = fluid.layers.concat(e,axis=1)
e = fluid.layers.reshape(e,(2,32,-1))
print('向量化拼接法2：不用循环拼接')
print(e.numpy())
'''
总结：
向量化过程确实与一般的reshape不同
拼接法2不用循环所以最佳
'''
```

# 解码器

 https://ai-studio-static-online.cdn.bcebos.com/72a0f0a9a4b64d3395a3eeeda888966214e61d30c0b8434b8376249f8131d9bc

解码器从正确的胶囊中接受一个16维向量，并学习将其解码为数字图像（它在训练时仅使用正确的胶囊向量，忽略不正确的）。解码器被用来作为正则子，它接受正确胶囊的输出作为输入，并学习重建一张28×28像素的图像，损失函数为重建图像与输入图像之间的欧氏距离。解码器强制胶囊学习对重建原始图像有用的特征。重建图像越接近输入图像越好。

下图是我自己训练的网络重构获得的图像，上面是输入网络的原图片，下面是网络输出rebuild的图片

 https://ai-studio-static-online.cdn.bcebos.com/047bbb59d3c74621a756a3f45048f0f83899bfd5fe4748a7b59713c734295acf



```python
#解压数据集文件
!unzip /home/aistudio/data/data10954/cat_12_test.zip 
!unzip /home/aistudio/data/data10954/cat_12_train.zip
```


```python
'''
@paddle实现完整代码 
@author Reatris 
使用minst数据集
'''
import numpy as np 
import paddle
import paddle.fluid as  fluid
import sys,os
import matplotlib.pylab as plt
from PIL import Image
import cv2

train_params = {
    'batch_size':32,
    'epoch_num':2,
    'save_model_name':'Capnet_class_and_rebuild',
    'm_plus':0.9,#计算损失值时大于这个值则loss=0
    'm_det':0.1,#计算损失值时大于这个值则loss=0
    'lambda_val':0.5,#错误概率loss权重
    'build_loss_scale':0.0005,#decode损失scale
    'with_decode':True,#是否训练decode部分
     'train_comparison':False,#是否使用CNN+POOL网络对比
     'challenge':False,#是否在训练过程中改变输入图片
     'train_data':'mnist',#'可以选择mnist'或者'cat12'数据集
}



class Capsule_Layer(fluid.dygraph.Layer):
    def __init__(self,pre_cap_num,pre_vector_units_num,cap_num,vector_units_num,use_Gaussian_distribution=False):
        '''
        胶囊层的实现类，可以直接同普通层一样使用
        Args:
            pre_vector_units_num(int):输入向量维度 
            vector_units_num(int):输出向量维度 
            pre_cap_num(int)：输入胶囊数 
            cap_num(int)：输出胶囊数 
            routing_iters(int):路由迭代次数，建议3次 
        Notes:
            胶囊数和向量维度影响着性能,可作为主调参数
        '''
        super(Capsule_Layer,self).__init__()
        self.use_Gaussian_distribution = use_Gaussian_distribution
        self.routing_iters = 3
        self.pre_cap_num = pre_cap_num
        self.cap_num = cap_num
        self.pre_vector_units_num = pre_vector_units_num
        self.vector_units_num = vector_units_num
        for j in range(self.cap_num):
            self.add_sublayer('u_hat_w'+str(j),fluid.dygraph.Linear(\
            input_dim=pre_vector_units_num,output_dim=vector_units_num))
    
    
    def squash(self,vector):
        '''
        压缩向量的函数，类似激活函数，向量归一化
        Args:
            vector：一个4维张量 [batch_size,vector_num,vector_units_num,1]
        Returns:
            一个和x形状相同，长度经过压缩的向量
            输入向量|v|（向量长度）越大，输出|v|越接近1
        '''
        vec_abs = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(vector)))  # 一个标量|v|模长度
        scalar_factor = fluid.layers.square(vec_abs) / (1 + fluid.layers.square(vec_abs))
        vec_squashed = scalar_factor * fluid.layers.elementwise_div(vector, vec_abs)  # 对应元素相除
        return(vec_squashed)

    def capsule(self,x,B_ij,j,pre_cap_num):
        '''
        这是动态路由算法的精髓。
        Args:
            x：输入向量,一个四维张量 shape = (batch_size,pre_cap_num,pre_vector_units_num,1)
            B_ij: shape = (1,pre_cap_num,cap_num,1)路由分配权重，这里将会选取(split)其中的第j组权重进行计算
            j：表示当前计算第j个胶囊的路由
            pre_cap_num:输入胶囊数
        Returns:
            v_j:经过多次路由迭代之后输出的4维张量（单个胶囊）
            B_ij：计算完路由之后又拼接(concat)回去的权重
        Notes:
            B_ij,b_ij,C_ij,c_ij注意区分大小写哦
        '''
        x = fluid.layers.reshape(x,(x.shape[0],pre_cap_num,-1))
        u_hat = getattr(self,'u_hat_w'+str(j))(x)
        u_hat = fluid.layers.reshape(u_hat,(x.shape[0],pre_cap_num,-1,1))
        shape_list = B_ij.shape#(1,1152,10,1)
        split_size = [j,1,shape_list[2]-j-1]#取出当前加权项
        for i in range(self.routing_iters):
            C_ij = fluid.layers.softmax(B_ij,axis=2)
            b_il,b_ij,b_ir = fluid.layers.split(B_ij,split_size,dim=2)#b_il是已经完成分配的项
            c_il,c_ij,b_ir = fluid.layers.split(C_ij,split_size,dim=2)
            v_j = fluid.layers.elementwise_mul(u_hat,c_ij)#加权[1,1152,1,1]x[32,1152,16,1]-->每16个向量分配一个权重
            v_j = fluid.layers.reduce_sum(v_j,dim=1,keep_dim=True)#分配到同一个路由的向量相加[32,1,16,1]
            v_j = self.squash(v_j)   #[32,1,16,1]
            v_j_expand = fluid.layers.expand(v_j,(1,pre_cap_num,1,1))#平铺[32,1,16,1]-->[32,1152,16,1]
            u_v_produce = fluid.layers.elementwise_mul(u_hat,v_j_expand)#求内积
            u_v_produce = fluid.layers.reduce_sum(u_v_produce,dim=2,keep_dim=True) #[32,1152,16,1]-->[32,1152,1,1]
            b_ij += fluid.layers.reduce_sum(u_v_produce,dim=0,keep_dim=True)#内积累加，更新路由权重BIJ
            B_ij = fluid.layers.concat([b_il,b_ij,b_ir],axis=2)
        return v_j,B_ij
    
    def forward(self,x):
        '''
        Args:
            x:shape = (batch_size,pre_caps_num,vector_units_num,1) or (batch_size,C,H,W)
                如果是输入是shape=(batch_size,C,H,W)的张量，
                则将其向量化shape=(batch_size,pre_caps_num,vector_units_num,1)
                满足:C * H * W = vector_units_num * caps_num
                其中 C >= caps_num
        Returns:
            capsules:一个包含了caps_num个胶囊的list
        '''
        
        if x.shape[3]!=1:
            x = fluid.layers.reshape(x,(x.shape[0],self.pre_cap_num,-1))
            temp_x = fluid.layers.split(x,self.pre_vector_units_num,dim=2)
            temp_x = fluid.layers.concat(temp_x,axis=1)
            x = fluid.layers.reshape(temp_x,(x.shape[0],self.pre_cap_num,-1,1))
            x = self.squash(x)
        B_ij = fluid.layers.ones((1,x.shape[1],self.cap_num,1),dtype='float32')/self.cap_num#路由分配权重[1,1152,10,1]
        capsules = []
        for j in range(self.cap_num):
            cap_j,B_ij = self.capsule(x,B_ij,j,self.pre_cap_num)#进行路由分配
            capsules.append(cap_j)
        capsules = fluid.layers.concat(capsules,axis=1)
        return capsules   

class Capconv_Net(fluid.dygraph.Layer):
    def __init__(self,train_data='mnist'):
        '''
        Args:
            decode_layer_num：decode部分的全连接层层数，用于遍历
            self.caps_0:这个是实例化了定义的Capsule_Layer,用法和其他层基本相同
        Notes:
            这里只用了一个胶囊层，不知道叠加层数会如何
        '''
        super(Capconv_Net,self).__init__()
        self.decode_layer_num = 3
        self.train_data = train_data
        if(train_data=='mnist'):
            self.add_sublayer('conv0',fluid.dygraph.Conv2D(\
            num_channels=1,num_filters=32,filter_size=(9,9),padding=0,stride = 1,act='relu'))
            self.add_sublayer('conv1',fluid.dygraph.Conv2D(\
            num_channels=32,num_filters=512,filter_size=(9,9),padding=0,stride = 2,act='relu'))
            self.caps_0 = Capsule_Layer(72,256,10,32)#胶囊层
            self.add_sublayer('rebuid_fc0',fluid.dygraph.Linear(input_dim=32,output_dim=512))
            self.add_sublayer('rebuid_fc1',fluid.dygraph.Linear(input_dim=512,output_dim=1024))
            self.add_sublayer('rebuid_fc2',fluid.dygraph.Linear(input_dim=1024,output_dim=784))
        else:
            self.conv0=fluid.dygraph.Conv2D(num_channels=3,num_filters=64,filter_size=(3,3),stride=1,padding=1,act='relu')
            self.pool0 = fluid.dygraph.Pool2D(pool_size=2,pool_stride=2,pool_type='max')#32,256
            self.conv1=fluid.dygraph.Conv2D(num_channels=64,num_filters=128,filter_size=(3,3),stride=1,padding=1,act='relu')
            self.pool1 = fluid.dygraph.Pool2D(pool_size=2,pool_stride=2,pool_type='max')#32,256
            self.conv2=fluid.dygraph.Conv2D(num_channels=128,num_filters=256,filter_size=(3,3),stride=1,padding=1,act='relu')
            self.pool2 = fluid.dygraph.Pool2D(pool_size=2,pool_stride=2,pool_type='max')#32,256
            self.caps_0 = Capsule_Layer(784,256,12,32)
            self.conv3=fluid.dygraph.Conv2D(num_channels=256,num_filters=512,filter_size=(3,3),stride=1,padding=1,act='relu')
            self.pool3 = fluid.dygraph.Pool2D(pool_size=2,pool_stride=2,pool_type='max')#32,256
            self.caps_1 = Capsule_Layer(196,512,12,32)#胶囊层
    def get_loss_v(self,label):
        '''
        计算边缘损失
        Args:
            label:shape=(32,10) one-hot形式的标签
        Notes:
            这里我调用Relu把小于0的值筛除
            m_plus：正确输出项的概率（|v|）大于这个值则loss为0，越接近则loss越小
            m_det：错误输出项的概率（|v|）小于这个值则loss为0，越接近则loss越小
            （|v|即胶囊(向量)的模长）
        '''
        max_l =  fluid.layers.relu(train_params['m_plus'] - self.output_caps_v_lenth)#广播
        max_l = fluid.layers.reshape(fluid.layers.square(max_l),(train_params['batch_size'],-1))#32,10
        max_r =  fluid.layers.relu(self.output_caps_v_lenth - train_params['m_det'])
        max_r = fluid.layers.reshape(fluid.layers.square(max_r),(train_params['batch_size'],-1))#32,10
        margin_loss = fluid.layers.elementwise_mul(label,max_l)\
                        + fluid.layers.elementwise_mul(1-label,max_r)*train_params['lambda_val']
        self.margin_loss = fluid.layers.reduce_mean(margin_loss,dim=1)

    def loss(self,label):
        '''
        计算边缘损失和decode
        Args:
            label:shape=(32,10) one-hot形式的标签
        Returns:
            total_loss or margin(不训练decode部分时)
        Notes:
            decode重建图像的损失用欧式距离计算
        '''
        self.get_loss_v(label)
        if train_params['with_decode']:
            euclidean_metric = fluid.layers.square(self.decode_result - self.input_imgs)#求得欧式距离
            self.reconstruction_err = fluid.layers.mean(euclidean_metric)
            self.total_loss = self.margin_loss + self.reconstruction_err * train_params['build_loss_scale']
            return self.total_loss #返回总loss
        else:
            return self.margin_loss #只返回边缘loss
            
        #只返回decode loss 冻结之后用
    
    def accuracy(self,label):
        label = fluid.layers.argmax(label,axis=1)
        T_L = fluid.layers.reshape(self.argmax_idx,(label.shape[0],))
        mask = paddle.fluid.layers.equal(T_L,label)
        mask = np.argwhere(mask.numpy()==True)
        return len(mask)/label.shape[0]
        

    def forward(self,x):
        if self.train_data == 'mnist':
            if train_params['with_decode']:
                self.input_imgs = fluid.layers.reshape(x,(train_params['batch_size'],-1))
            x = self.conv0(x)
            x = self.conv1(x)       
            self.output_caps = self.caps_0(x)# 32,10,16,1
            self.output_caps_v_lenth = fluid.layers.sqrt(fluid.layers.reduce_sum(\
            fluid.layers.square(self.output_caps),dim=2,keep_dim=True))##计算模长32,10,1,1
            softmax_v = fluid.layers.softmax(self.output_caps_v_lenth,axis=1)#计算每个胶囊类概率 32,10,1,1
            argmax = fluid.layers.argmax(softmax_v,axis=1)#选取最大概率索引32,1,1
            mask_v = []
            self.argmax_idx = fluid.layers.reshape(argmax,(argmax.shape[0],1))
            
            if train_params['with_decode']:
                ######下面是decode部分######
                #选取正确胶囊
                for i in range(train_params['batch_size']):
                    v = self.output_caps[i][self.argmax_idx[i],:]
                    mask_v.append(fluid.layers.reshape(v,(1,self.output_caps.shape[2])))
                self.mask_v = fluid.layers.concat(mask_v,axis=0)#32,16
                #全连接层重构图像
                x = self.mask_v
                for i in range(self.decode_layer_num):
                    x = getattr(self,'rebuid_fc'+str(i))(x)
                self.decode_result = x
        else:
            capsule_list = []
            for i in range(4):
                x = getattr(self,'conv'+str(i))(x)
                x = getattr(self,'pool'+str(i))(x)
                if i>1:
                    capsule_list.append(x)
            output_cap_list = []
            for j in range(2):
                # print(capsule_list[j+1].shape)
                output_cap_list.append(getattr(self,'caps_'+str(j))(capsule_list[j]))
            self.output_caps = fluid.layers.concat(output_cap_list,axis=2)
            # print(self.output_caps.shape)
            self.output_caps_v_lenth = fluid.layers.sqrt(fluid.layers.reduce_sum(\
            fluid.layers.square(self.output_caps),dim=2,keep_dim=True))##计算模长32,10,1,1
            softmax_v = fluid.layers.softmax(self.output_caps_v_lenth,axis=1)#计算每个胶囊类概率 32,10,1,1
            argmax = fluid.layers.argmax(softmax_v,axis=1)#选取最大概率索引32,1,1
            mask_v = []
            self.argmax_idx = fluid.layers.reshape(argmax,(32,1))
            
#用同规模卷积神经网络+池化层对比
class Mnistnet(fluid.dygraph.Layer):
    def __init__(self,train_data='mnist'):
        super(Mnistnet,self).__init__()
        self.train_data = train_data
        if(train_data=='mnist'):
            self.add_sublayer('conv0',fluid.dygraph.Conv2D(\
            num_channels=1,num_filters=32,filter_size=(9,9),padding=0,stride = 1,act='relu'))
            self.add_sublayer('conv1',fluid.dygraph.Conv2D(\
            num_channels=32,num_filters=512,filter_size=(9,9),padding=0,stride = 2,act='relu'))
            self.pool = fluid.dygraph.Pool2D(global_pooling=True,pool_type='max')#32,256
            self.fc0=fluid.dygraph.Linear(input_dim=512,output_dim=128,act='relu')#[32,256] ==> [32,50]
            self.fc2=fluid.dygraph.Linear(input_dim=128,output_dim=10,act='softmax')
        else:
            self.block1_conv1_3_64=fluid.dygraph.Conv2D(num_channels=3,num_filters=64,filter_size=(3,3),stride=1,padding=1,act='relu')
            self.pool0 = fluid.dygraph.Pool2D(pool_size=2,pool_stride=2,pool_type='max')#32,256
            self.block1_conv2_3_64=fluid.dygraph.Conv2D(num_channels=64,num_filters=128,filter_size=(3,3),stride=1,padding=1,act='relu')
            self.pool1 = fluid.dygraph.Pool2D(pool_size=2,pool_stride=2,pool_type='max')#32,256
            self.block2_conv1_3_128=fluid.dygraph.Conv2D(num_channels=128,num_filters=256,filter_size=(3,3),stride=1,padding=1,act='relu')
            self.pool2 = fluid.dygraph.Pool2D(pool_size=2,pool_stride=2,pool_type='max')#32,256
            self.block2_conv2_3_128=fluid.dygraph.Conv2D(num_channels=256,num_filters=512,filter_size=(3,3),stride=1,padding=1,act='relu')
            self.g_poo1 = fluid.dygraph.Pool2D(global_pooling=True,pool_type='max')
            self.fc0 = fluid.dygraph.Linear(input_dim=896,output_dim=256,act='relu')
            self.fc1 = fluid.dygraph.Linear(input_dim=256,output_dim=12,act='softmax')
    def forward(self,x):
        if self.train_data =='mnist':
            for layers in self.sublayers():
                x = layers(x)
                x = fluid.layers.squeeze(x,axes=[])
            return x
        else:
            x =self.block1_conv1_3_64(x)
            x = self.pool0(x)
            x = self.block1_conv2_3_64(x)
            x = self.pool1(x)
            f1 =  fluid.layers.squeeze(self.g_poo1(x),axes=[])
            x = self.block2_conv1_3_128(x)
            x = self.pool2(x)
            f2 = fluid.layers.squeeze(self.g_poo1(x),axes=[])
            x = self.block2_conv2_3_128(x)
            f3 = fluid.layers.squeeze(self.g_poo1(x),axes=[])
            x = fluid.layers.concat((f1,f2,f3),axis=1)
            x = self.fc0(x)
            x = self.fc1(x)
            return x
   


#猫12分类数据读取
def data_load(train_list_path,batch_size):
    '''
    train_list_path:标注文件txt所在path
    '''
    train_dir_list=[]
    train_label=[]
    with open(train_list_path,'r') as train_dirs:
        #train_dir_list.append(train_dirs.readline())
        lines=[line.strip() for line in train_dirs]
        for line in lines:
            img_path,label=line.split()
            train_dir_list.append(img_path)
            train_label.append(label)
    def reader():
        imgs=[]
        labels=[]
        img_mask=np.arange(len(train_dir_list)) #生成索引
        np.random.shuffle(img_mask) #随机打乱索引
        count=0
        for i in img_mask:
            img=cv2.imread(train_dir_list[i])
            img=cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)/255
            img=np.transpose(img,(2,0,1))
            imgs.append(img)
            temp_labels_one_hot = np.zeros(12)
            temp_labels_one_hot[int(train_label[i])] = 1 
            labels.append(temp_labels_one_hot)
            count+=1
            if(count%train_params['batch_size']==0):
                yield np.asarray(imgs).astype('float32'),np.asarray(labels).astype('float32').reshape((train_params['batch_size'],12))
                imgs=[]
                labels=[]
    return reader


def draw_train_process(iters,accs_mnist,loss_mnist,accs,loss):
    '''
    训练可视化(对比)
    '''
    plt.title('CapsulesNet training',fontsize=24)
    plt.xlabel('iters',fontsize=20)
    plt.ylabel('acc/loss',fontsize=20)
    plt.plot(iters,loss_mnist,color='yellow',label='loss_of_cnn+pool')
    plt.plot(iters,accs_mnist,color='blue',label='accuracy_cnn+pool')
    plt.plot(iters,accs,color='red',label='accuracy_capsules_net')
    plt.plot(iters,loss,color='green',label='loss_of_capsule_net')
    plt.legend()
    plt.grid()
    plt.show()

def draw_train_process2(iters,accs,loss):
    '''
    训练可视化
    '''
    plt.title('CapsulesNet training',fontsize=24)
    plt.xlabel('iters',fontsize=20)
    plt.ylabel('acc/loss',fontsize=20)
    plt.plot(iters,accs,color='red',label='accuracy')
    plt.plot(iters,loss,color='green',label='loss')
    plt.legend()
    plt.grid()
    plt.show()

def show_array2img(array,title):
    rebuilded_img = Image.fromarray(array.astype('uint8')).convert('RGB')
    plt.imshow(rebuilded_img)
    plt.title(title)
    plt.show()

with fluid.dygraph.guard():
    if train_params['train_data']=='mnist':    
        train_reader = paddle.batch(reader=paddle.reader.shuffle(\
        paddle.dataset.mnist.train(),buf_size=512),batch_size=train_params['batch_size'])  
    else:
        train_reader = data_load('train_list.txt',train_params['batch_size'])
    if train_params['train_comparison']:
        train_params['with_decode']=False#对比时就不训练decode部分了
        print('start training CNN+POOL net')
        model_mnist=Mnistnet(train_params['train_data'])
        model_mnist.train()
        if train_params['train_data']=='mnist':
            opt_mnist=fluid.optimizer.AdamOptimizer(learning_rate=0.01,parameter_list=model_mnist.parameters())
            checkpoint=train_params['epoch_num']*1876*0.5
        else:
            opt_mnist=fluid.optimizer.AdamOptimizer(learning_rate=0.001,parameter_list=model_mnist.parameters())
            checkpoint=train_params['epoch_num']*68*0.5
        epoch_num_mnist = train_params['epoch_num']
        all_train_costs_mnist = []
        all_train_accs_mnist = []
        all_iter = 0
        for pass_num_mnist in range(epoch_num_mnist):
            for batch_id_mnist,data in enumerate(train_reader()):
                if train_params['train_data']=='mnist':
                    temp_images = []
                    temp_labels = []
                    for i in range(32):
                        if(all_iter>checkpoint and train_params['challenge']):
                            temp_images.append(np.transpose(np.reshape(data[i][0],(1,28,28)),(0,2,1)))
                        else: 
                            temp_images.append(np.reshape(data[i][0],(1,28,28)))
                        temp_labels.append(data[i][1])
                    temp_images=fluid.dygraph.to_variable(np.asarray(temp_images).reshape((32,1,28,28)))
                    temp_labels=fluid.dygraph.to_variable(np.asarray(temp_labels).reshape((32,1)))
                else:
                    temp_images,temp_labels = data
                    if(all_iter>checkpoint and train_params['challenge']):
                        temp_images=fluid.dygraph.to_variable(np.transpose(\
                        np.asarray(temp_images).reshape((32,3,224,224)),(0,1,3,2)))
                    else:
                        temp_images=fluid.dygraph.to_variable(np.asarray(temp_images).reshape((32,3,224,224)))
                    temp_labels=fluid.dygraph.to_variable(np.asarray(temp_labels).reshape((32,12)))
                    temp_labels = fluid.layers.reshape(fluid.layers.argmax(temp_labels,axis=1),(32,1))

                predict = model_mnist(temp_images)
                #计算loss
                loss_mnist= fluid.layers.cross_entropy(label=temp_labels,input=predict)
                    
                #计算accuracy
                acc_mnist=fluid.layers.accuracy(input=predict,label=temp_labels)
                avg_loss_mnist = fluid.layers.mean(loss_mnist)
                avg_loss_mnist.backward()#反向传播跟新weights
                opt_mnist.minimize(avg_loss_mnist)
                opt_mnist.clear_gradients()
                all_iter +=1
                if all_iter%100==0:
                    all_train_costs_mnist.append(avg_loss_mnist.numpy()[0])
                    all_train_accs_mnist.append(acc_mnist.numpy()[0])
                    print('pass_num:{},iters:{},loss:{},acc:{}'.format(\
                    pass_num_mnist,all_iter,avg_loss_mnist.numpy()[0],acc_mnist.numpy()[0]))
        print("Final loss of cnn+pool: {}".format(avg_loss_mnist.numpy()))
    
    print('start training CNN+Capsules')
    model = Capconv_Net(train_params['train_data'])  #实列化模型
    if os.path.exists(train_params['save_model_name'] + '.pdparams') and not train_params['train_comparison']:#存在模型参数则继续训练
        print('continue training')
        param_dict,_ = fluid.dygraph.load_dygraph(train_params['save_model_name'])
        model.load_dict(param_dict)
    model.train()
    all_iter = 0
    all_loss = []
    all_iters = []
    all_accs = []
    if train_params['train_data']=='mnist':
        opt=fluid.optimizer.AdamOptimizer(learning_rate=0.01,parameter_list=model.parameters())
        checkpoint=train_params['epoch_num']*1876*0.5
    else:
        opt=fluid.optimizer.AdamOptimizer(learning_rate=0.001,parameter_list=model.parameters())
        checkpoint=train_params['epoch_num']*68*0.5
    for pass_num in range(train_params['epoch_num']):
        for pass_id,data in enumerate(train_reader()):
            if train_params['train_data']=='mnist':
                temp_images = []
                temp_labels = []
                for i in range(32):
                    if(all_iter>checkpoint and train_params['challenge']):
                        temp_images.append(np.transpose(np.reshape(data[i][0],(1,28,28)),(0,2,1)))
                    else: 
                        temp_images.append(np.reshape(data[i][0],(1,28,28)))
                    temp_labels_one_hot = np.zeros(10)
                    temp_labels_one_hot[data[i][1]] = 1 
                    temp_labels.append(temp_labels_one_hot)
                temp_images = fluid.dygraph.to_variable(np.asarray(temp_images))#转换成tensor才能输入
                temp_labels = fluid.dygraph.to_variable(\
                np.asarray(temp_labels).reshape((32,10)).astype('float32'))
            else:
                temp_images,temp_labels = data
                if(all_iter>checkpoint and train_params['challenge']):
                    temp_images=fluid.dygraph.to_variable(np.transpose(\
                    np.asarray(temp_images).reshape((32,3,224,224)),(0,1,3,2)))
                else: 
                    temp_images=fluid.dygraph.to_variable(np.asarray(temp_images).reshape((32,3,224,224)))
                
                temp_labels=fluid.dygraph.to_variable(np.asarray(temp_labels).reshape((32,12)))
            model(temp_images)
            loss = model.loss(temp_labels)
            avg_loss=fluid.layers.mean(loss)
            avg_loss.backward()
            opt.minimize(avg_loss)
            opt.clear_gradients()
            all_iter+=1
            if all_iter%100==0:
                acc=model.accuracy(temp_labels)
                all_loss.append(avg_loss.numpy()[0])
                all_iters.append(all_iter)
                all_accs.append(acc)
                print('pass_epoch:{},iters:{},loss：{},acc：{}'.format(pass_num,all_iter,avg_loss.numpy()[0],acc))
    fluid.save_dygraph(model.state_dict(),train_params['save_model_name']) #保存模型参数
    if train_params['train_comparison']:
        draw_train_process(all_iters,all_train_accs_mnist,all_train_costs_mnist,all_accs,all_loss)
    else:
        draw_train_process2(all_iters,all_accs,all_loss)
    print('finished training')
    if train_params['with_decode']:
        #图像重构显示
        show_array2img(np.reshape(model.input_imgs[0].numpy(),(28,28)),'imput_img')
        show_array2img(np.asarray(np.reshape(model.decode_result[0].numpy(),(28,28))),'rebuild_img')
   
        
```

    start training CNN+Capsules
    pass_epoch:0,iters:100,loss：0.04529523849487305,acc：0.4375
    pass_epoch:0,iters:200,loss：0.025471236556768417,acc：0.90625


# 性能评估

**说了这么多胶囊神经网络性能到底如何呢，让我们用同规模CNN+最大池化层来对比下对比一下**

 https://ai-studio-static-online.cdn.bcebos.com/aa8bc4d29c8246cf9c12113174893f58aa71e06b16894ee894dba4434debedb9

这是两个网络在其他条件相同情况下进行的1800次迭代，一开始胶囊神经网络起步慢，但是结果很明显胶囊神经网络更加稳定，CNN+池化层准确率波动不小

**再来试一下，当训练到一半时将所有输入网络的图片转置（你可以理解为将数字翻转，改变位姿）的情况**

 https://ai-studio-static-online.cdn.bcebos.com/0181eee5f2d9445da1c58316e99f663a180bdaeacbd945f7ba0c87b268c8127f


* 可以明显的看到CNN+池化层在图片转置的情况下准确率直接跌落谷底，在之后的训练中也是一蹶不振（迷失了自我）！

* 但是胶囊神经网络就不一样了，面对截然不同的图片仍然有高于50%的正确率，而且在之后迅速恢复了100%的正确率！甩了CNN+池化层一大截！Capsule显露了它处理不同位姿的本领

**胶囊数量和向量维度对性能的影响**
 https://ai-studio-static-online.cdn.bcebos.com/d6324887d5654a7aa3a55a169e1578ff6df1fa73ce5a49eca5f0682ea62d5fcf


# 总结

**1.路由迭代次数设置3最佳，次数过多效果并不好（针对这个项目）**

**2.仅仅只需要迭代300次，分类结果几乎100%正确，可见胶囊神经网络的强大**

**3.当acc达到1时，可以冻结前面的参数，训练decode部分**

**4.Capsule对于位姿的理解真实存在**

**5.原理理解十分重要，要把原理理解透彻哦**

最后感谢AI Stiudio社区提供的平台 以及一起在平台学习的划桨手们，让我们一起进步

2020年7月 AI Stiudio社区  

关于作者


| Reatris| 计算机专业 大三 |
| -------- | -------- | -------- |
| 研究方向 | 主攻计算机视觉，玩转动态图,我会不定期更新我的项目,欢迎大家fork、评论、喜欢三连 |
| 主页|[https://aistudio.baidu.com/aistudio/personalcenter/thirdview/206265](http://aistudio.baidu.com/aistudio/personalcenter/thirdview/206265)|


和我一起来玩转paddle 2.0 动态图吧~


```python

```
