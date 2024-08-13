# plonk
本文学习plonk。参考：

[zkSnark-Plonk证明系统](https://drive.google.com/file/d/1-wcIHORaRMkcZX32YBey2boUF5d5Aw1t/view?usp=sharing)
[TOC]

# 简介
Plonk系统包括电路系统和多项式系统，怎么构造Plonk标准门以及验证方怎么验证标准门

# 密码学承诺
**Schwartz-Zippel引理**：令P为有限域$\mathbb F$上的多项式$P = F(x_1, x_2, \ldots, x_n)$, 其阶为d。令S为有限域$\mathbb F$的子集，从S中选择随机数$r_1, r_2\ldots, r_n$，则多项式等于零的概率可以忽略，即：

$$
\Pr[P(r_1, r_2, \ldots, r_n) = 0] \leq \frac{d}{|S|}
$$
单变量的情况下，等价于多项式的阶为d，则最多有d个根。

## 承诺的概念
* 承诺：选择x，计算$y=f(x)$，发送y。
* 完全打开，发送原像x。
* 校验$x = f(y)$
* 打开n个随机点。
* 校验n个随机点（概率上验证）

对函数的要求：
* 函数求逆是NP困难的
* 校验简单
* 通常加密货币使用145步代替123步，用概率来保障多项式安全，只有知道多项式的具体解才能打开n个点。可以节省验证复杂度。

## 哈希承诺
* 广播哈希值y
* 完全打开：广播x
* 验证$y = HASH(x)$

## merkle 承诺与Merkle证明
* 承诺：发送Merkle Root
* 打开一个随机点x和x的路径path
* 通过x，root，path校验$root = Merkle(x,path)$

比较低效的做法是直接打开所有点，每个验证来保证每个节点都是对的，但是过于耗费算力。高效的做法是选择检测n个点即可，概率的角度就难以伪造。

Merkle证明就是当我拥有一个交易的私钥并发送交易给别人，通过交易和交易的路径可以计算出一个Root并与以太坊上的root比较。如此证明一个人有花费某token的权力。

## Sigma零知识证明
知道秘密$\omega$，且与公开输入的Q满足$Q=\omega\times G$
* 承诺：选择随机数r，计算发送承诺$A = r\times G$
* 挑战：验证者发送随机数e
* 响应：计算$z = r + e\times\omega$，发送z
* 验证：校验$z\times G = A  + e\times Q$

是一个交互式零知识证明，随机数r遮蔽$\omega$但是仍然能证明z里面包含了关于$\omega$的知识。在实践中可以使用Q和E的哈希值来代替随机挑战e，来保证无交互。

## Pedersen承诺

初始化：椭圆曲线生成元G，H，标量域$F_r$(有限域)，基域$F_q$(椭圆曲线点)
* 承诺：对金额m和随机数r，计算$P = mG+rH$
* 完全打开：发送m和r
* 校验$P=mG+rH$

如果某个账户知道一个以太坊树节点上的余额和随机数r就可以验证承诺取款。这个承诺具有同态性：
* 用户1，2的承诺是$(m_1, r_1),(m_2, r_2)$。两个用户向Alice支付余额并告诉随机数
* 矿工更新Alice的余额承诺$P = P_1+P_2$
* 由于Alice知道$m_1+m_2$和$r_1+r_2$，于是可以验证承诺P

这就使Tornado cash的原理。

### 后门
如果Alice知道G和H之间的离散对数$H = \alpha G$，可以声称自己有大金额：
$$
r' = r-(m'-m)\alpha^{-1}
$$

# 多项式承诺
* 承诺：选择x，计算$y = f(x)$，发送y
* 完全打开x
* 校验：$y = f(x)$
* 打开n个随机点
* 校验n个随机点

## 困难假设
* 离散对数困难假设：椭圆曲线群和生成元$\mathbb G*, G$，已知$G,\alpha G$，任意多项式算出$\alpha$的可能性可忽略。
* t阶强Diffie-Hellman假设：公开$PK = \{\alpha^i G\}_{i = 0}^t \in \mathbb G^{ (t+1) }$。对任意值c能计算出$\frac{1}{\alpha+c}G$的改了吧可以忽略。
* Q阶离散对数假设：公开群$\mathbb G_1 \mathbb G_2$，通过$PK = \{\alpha^i G_1,\alpha^i G_2\}_{i = 0}^Q$可以算出$\alpha$的概率忽略不计。

## 多项式承诺定义
承诺、打开、校验、打开随机点、验证随机点。一般用45步代替23步在概率上保证作弊概率可忽略。
* 初始化群$\mathbb G*$和用于承诺的公钥$PK(\alpha)$，$\alpha$要删除。
* 承诺：输入公钥PK和多项式$\phi(x)$，输出承诺C
* 完全打开：输出多项式$\phi(x)$的系数（数据量太大）
* 校验：验证承诺C与多项式$\phi(x)$一致（验证复杂度高）
* 打开一个随机点i，输出$(i, \phi(i), w_i)$, $w_i = \frac{\phi(x)-\phi(i)}{x-i}$
* 校验随机点

核心思想：以下五个描述等价
* 数据多项式与电路多项式满足运算关系
* 多项式正确
* 几个随机打开点正确
* 商多项式存在
* 验证方计算双线性映射成立。

## 多项式承诺性质
一致性
* 完全打开一致性：能通过完全打开的信息$\phi(x)$和承诺C验证。
* 随机打开点一致性：可以通过$(i, \phi(i), w_i)$和承诺C验证。

**多项式绑定**：对于承诺C，攻击者找到两个不同的多项式$\phi(x),\phi'(x)$使得都能通过C校验的成功概率可忽略。

**随机打开点绑定性**：对于承诺C和位置i攻击者可以输出两个多项式值$\phi(i),\phi'(i)$都能通过校验的概率可忽略。

**保密性**：对于多项式$\phi(x)$，已知$PK,C,(i, \phi(i), w_i)$，任意多项式实践内攻击者不能从其他点得到多项式值$\phi(i')$

从加密的角度讲可以把公钥加密对应到承诺(PK加密)，把数字签名对应到zk的秘密知识(sk)

# KZG多项式承诺
**系统初始化**；对于椭圆曲线双线性群$\mathcal G = (e, \mathbb G, \mathbb G_T),$有毒废料$\alpha \in_R \mathbb Z_p*$。输出为：
$$
PK = (\mathcal G, G, \alpha G \ldots, \alpha^t G)
$$
setup分为两个集合，一个是PK，一个是VK。两个集合可以是重叠关系。这部分术语叫common reference string CRS，有时也叫struct reference string SRS。
## 1个多项式打开1个随机点
**承诺：**对于多项式$\phi(x) = \sum \phi_j\times x^j$，计算承诺：
$$
C = \phi(\alpha)G = \sum \phi_j  \alpha^j G
$$

**完全打开**：输出多项式系数。

**验证**：验证
$$
C' = \phi(\alpha)G = \sum \phi_j  \alpha^j G = C
$$

在实际上完全打开需要很长的VK（t+1个）和很多的验证计算（计算t+1次椭圆曲线点加）。会消耗大量gas，不值得。

**打开一个随机点**：计算商多项式：
$$
\psi_i(x) = \frac{\phi(x) - \phi(i)}{x-i}
$$
基于商多项式的系数和PK，计算商多项式承诺：
$$
W_i = \psi_i(\alpha)G
$$

输出$(i, \phi(i), W_i)$

**校验一个随机点：**：校验等式：
$$
e(C, G) = e(W_i, \alpha G-i G)\times e(G,G)^{\phi(i)}
$$
承诺$VK = (G, \alpha G)$，这样VK就很小了。

### 关键结论
* 充要条件1：商多项式存在（除得尽）等价于上面的双线性映射成立
* 充要条件2：
    * 商多项式存在等价于$(i, \phi(i))$正确。
    * 商多项式不存在等价于$(i, \phi(i))$错误。

商多项式是KZG的核心，所以除得尽是一个构造承诺的重要性质。
## t个多项式打开1个随机点
设我们的多项式为$\phi_i(x)$，这里承诺计算所有多项式的承诺，记为$C_i$。完全打开仍然是计算不经济的。在打开验证随机点的时候不是一个一个多项式计算而是计算一个聚合的多项式：使用transcript计算随机数$\gamma$
$$
\psi_z(x) = \sum_{i=1}^t \gamma^{t-1} \frac{\phi_i(x)-\phi_i(z)}{x-z}
$$
商多项式承诺为：
$$
W_z = \psi_z(\alpha) G_1
$$
验证时验证：
$$
F = \sum_{i = 1}^t \gamma^{i-1}C_1\quad V = \sum_{i = 1}^t \gamma^{i-1} \phi_i(z) G_1
$$
推导公式成立：
$$
e(F-V, G_2)\times e(-W_z,\alpha G_2-zG_2) = 1
$$

这里区别在于使用了同一个验证$W_z$来代替对每个不等式生成W。这样的好处是节省来证明难度和传输消耗。这里根据S-Z引理认为两个不同的不等式在随机点碰撞的概率可以忽略不计。

## t个多项式打开多个随机点
此时把多项式分为多组计算，比如对于两个点就分为两组。对于多个随机点计算多个$\psi_z(x)$，其他计算相同。接着对两组多项式分别求：
$$
F = F_1-V_1 + r(F_2-V_2)
$$

接着验证：
$$
e(F+z_1\times W_{z_1} + rz_2\times W_{z_2}, G_2)\times e(-W_{z_1}-r\times W_{z_2}, \alpha \times G_2) = 1_{G_T}
$$

## 总结
虽然算法对多多项式和多取值有一定优化但是可以看见KZG多项式越多，打开点越多验证越复杂。后面介绍的Dan Boneh承诺验证复杂度仅与多项式个数有关而与随机打开点数无关。

# Dan Boneh 承诺（Shplonk）
高阶多项式$f$在集合$S\subset\mathbb F$上的函数值等于的姐多项式$r$在$z\in S$上的点函数值$r(z) = f(z)$

## 关键结论
* 对于$|S|$个打开点，两个多项式的值相等：$r(z) = f(z)$
* 目标多项式$Z_S(X) = \Pi_{z\in S}(X-z)$是多项式$f(x)-r(x)$的引子。
* 条件1与条件2等同，$Q(x) = \frac{f(X)-r(X)}{Z_S(X)}$成为商多项式，分母成为目标多项式。

可以看出相对于打开多个点验证多项式，构造商多项式验证复杂度仅和多项式个数有关。
## t打开s个点
**系统初始化**：双线性群为$\mathcal G = (e,\mathbb G_1, \mathbb G_2, \mathbb G_3)$，有毒废料$\alpha$，d+t元组

$<G_1, \alpha G_1, \ldots, \alpha^{d-1}G_1;G_2,\alpha G_2,\ldots,\alpha^tG_2>\in(\mathbb G_1^d,\mathbb G_2^t)$，输出为：
$$
PK = (\mathcal G, G_1, \alpha G_1, \ldots, \alpha^{d-1}G_1;G_2,\alpha G_2,\ldots,\alpha^tG_2)
$$

**多项式承诺**：对每个多项式求，与上面一样。

**打开随机点**：计算随机数$\gamma$，计算所有打开点的商多项式：
$$
h(X) = \sum_{i\in [t]} \gamma^{i-1} \frac{f_i(X) - r_i(X)}{Z_{S_i}(X)}
$$
并只发送一个商多项式承诺：$W = h(\alpha)\times G_1$
**验证随机点**：验证成立：
$$
\Pi_{i\in[t]} e(\gamma^{i-1}\times(C_i - r_i(\alpha)\times G_1), Z_{T/S_i}(\alpha) \times G_2) = e(W, Z_T(\alpha)\times G_2)
$$
这里把本来在右边的$e(W,Z_{S_i}(\alpha)\times G_2)$挪到左边共同计算节省了一些双线性计算。

## t个多项式打开s个随机点2.0
为了进一步简化我们有第三个同等命题：
$$
Z_S(X)|g(X) \leftrightarrow Z_T(X) | Z_{T/S}(X)\times g(X)
$$
**系统初始化**：双线性群$\mathcal G = (e, \mathbb G_1, \mathbb G_2, \mathbb G_T)$，随机数$\alpha \in_R \mathbb Z_p*$，d+2元组 $<G_1, \alpha G_1, \ldots, \alpha^{d-1}G_1; G_2, \alpha G_2> \in (\mathbb G_1^d, \mathbb G_2^2)$，令输出为
$$
<\mathcal G, G_1, \alpha G_1, \ldots, \alpha^{d-1}G_1; G_2, \alpha G_2> 
$$
**承诺**：同上，对于每个多项式计算

**打开随机点**：使用上面的同等计算h：
$$
h(X) = \frac{\sum_{i\in [t]}\gamma^{i-1}Z_{T/S_i}(X)\times (f_1(X) - r_1(X))}{Z_T(X)}
$$
发送一个商多项式承诺
$$
W_1 = h(\alpha)G_1
$$
*这部分其实一样*，接着通过上面多项式计算随机数z：

**区别点：计算辅助多项式：**
$$
f_z(X) = \sum_{i\in [k]}\gamma^{i-1} \times Z_{T/S_i}(z)\times (f_i(X)-r_i(z))
$$
$$
L(X) = f_z(X) - Z_T(z)\times h(X)
$$
由于$L(z) = f_z(z) - Z_T(z)\times h(z)$，所以$(X-z)|L(X)$

**区别点：计算发送辅助多项式的商多项式承诺：**

$$
W_2 = \frac{L(\alpha)}{\alpha - z}\times G_1
$$
发送$W_1$ 和 $W_2$。相比上面的算法增加了一个证明。验证：

$$
e(\sum_{i\in[t]}\gamma^{i-1}\times Z_{T/S_i}\times (C_i - r_i(z)\times G_1) - Z_T(z) \times W_1, G_2) = e(W_2,(\alpha - z)\times G_2)
$$

这部分优化其实是因为上面的双线性配对两边都有i，所以无法把双线性配对外的累加放到双线性配对里。而通过添加一个证明让e内部两边都有i变成等式两边的e内部都有i，让累加可以放入括号。于是双线性配对的复杂度变成椭圆曲线累加的复杂度，这样就大大减少了计算量。

## t-s3.0
把上节中验证公式的$r_i(z)$项提出，因为整个项变量只有z所以可以直接先求和再求椭圆曲线，减少计算量。
## t-s4.0
把$W_2 = \frac{L(\alpha)}{\alpha - z}G_1$变为$W_2 = \frac{L(\alpha)}{Z_{T/S_1}(z)( \alpha - z )}G_1$。在k=1的时候可以再简化为：
$$
F = \frac{-Z_T(z)}{Z_{T/S_1}(z)}W_1 + C - r_1(z)G_1
$$
这样只有三个倍点运算。问题在于Plonk有多个多项式，下面介绍怎么把多个多项式单点打开等价转化为1个多项式。
# Fflonk
核心思想为把n个简单的多项式转换为一个等价表达的多项式g。
## 定义
t个多项式的组合$combine_t(\bar f): \mathbb F[X]^t \rightarrow \mathbb F[X]$：
$$
g(X) = \sum_{i < t}f_i(X^t)X^i
$$
同样通过分别取对应位置的系数可以分解多项式。类似FFT的思想。这样转换之后通过一个多项式就可以使用t-s4.0的算法，将计算量减少。

***这里原文给的不是很清楚，待补充***

# Plonk证明系统
## 电路化：门和线
Groth通过电路约束来表达电路，但是Plonk门因为比较灵活没法表达庞大的电路（需要很多约束）。下面是Plonk门的具体构造
### 门约束
创建一个通用方程：
$$
Q_{Li}\times a_i + Q_{Ri}\times b_i + Q_{Oi}\times c_i + Q_{Mi} \times a_i \times b_i + Q_{Ci} = 0
$$
通过选取不同的参数可以模拟不同门。通用方程可以表达加法门、乘法门、常量门、加法门与乘法门的耦合。这种一门一个多项式会造成过多的离散值，所以在有多个门的时候可以通过多项式进行封装。对于每个门的系数可以得出多项式点值，比如对加法门可以获得关于$Q_L$的点值，通过FFT得到系数值，就得到了每个门系数和三个输入输出值的多项式表达。可以得到一个通用表达式：
$$
Q_L(x)\times a(x) + Q_R(x)\times b(x) + Q_o(x)\times c(x) + Q_M(x)\times a(x) \times b(x) + Q_c(x) = Z(x)\times H(x)
$$
这个通用表达式在几个固定值（与Groth16类似的思路）会取零，所以可以被目标多项式整除。这样就构造出可以放入前面讨论的多项式证明的多项式了。

### 线约束
在庞大电路中，总有一些输入输出值是相同的，即线约束。Plonk引入坐标对累加器来表示这种约束。

给定多项式$P(X)$的值表达，横坐标$x=(0, 1, 2, 3)$，纵坐标$y = (-1, 1, 0, 1)$。可以得到基于x的X系数表达：
$$
X(x) = x
$$
以及通过FFT获得Y系数表达：
$$
Y(x) = x^3 - 5x^2 + 7x -2
$$
**坐标累加器P(x)的递归表达：**

$$
P(n+1) = P(n)\times (u + X(n) + v \times Y(n))\quad P(0) = 1
$$
其中uv是随机常量。可以得到通项表达：
$$
P(n) = \Pi_{i = 0}^{n-1} (u + X(i) + v\times Y(i))
$$
由于纵坐标2，4大小相同，根据乘法交换律可以交换x为$(0, 3, 2, 1)$。有两个关键结论：
* 改变横坐标索引，如果纵坐标相等，则坐标累加器不变。
* 改变横坐标索引，如果坐标累加器相等，则两个纵坐标相等，确保线约束成立。

所以这时只需要广>播两个坐标累加器就可以证明两个index上y相等。

### 约束汇总
结合前面的门约束：
$$
Q_L(x)\times a(x) + Q_R(x)\times b(x) + Q_o(x)\times c(x) + Q_M(x)\times a(x) \times b(x) + Q_c(x) = Z(x)\times H(x)
$$
加上线约束：
$$
P_a(\omega x) - P_a(x)(u + x + va(x)) = Z(x)H_1(x)
$$
$$
P_{a'}(\omega x) - P_{a'}(x)(u + \sigma_a(x) + va(x)) = Z(x)H_2(x)
$$
$$
P_b(\omega x) - P_b(x)(u + g\times x + vb(x)) = Z(x)H_3(x)
$$
$$
P_{b'}(\omega x) - P_{b'}(x)(u + \sigma_b(x) + vb(x)) = Z(x)H_3(x)
$$
$$
P_c(\omega x) - P_c(x)(u + g^2\times x + vc(x)) = Z(x)H_5(x)
$$
$$
P_{c'}(\omega x) - P_{c'}(x)(u + \sigma_c(x) + vc(x)) = Z(x)H_6(x)
$$
结合两者可以规定所有约束
## Plonk证明系统
这段比较复杂，可以见[ 参考文件 ](https://drive.google.com/file/d/1-wcIHORaRMkcZX32YBey2boUF5d5Aw1t/edit)

* 为了确保零知识，红色部分是随机项
* 商多项式规模比较大所以分成三个承诺

# 聚合证明系统
对于多组算法设计一个电路并只通过验证一个电路校验所有电路。可以把所有电路的最后双线性映射前的值通过随机数组合起来（类似多个多项式展开聚合），接着放入双线性映射校验。所以这个聚合证明电路就是一个双线性映射，通过验证这个双线性映射电路就可以验证所有证明。

这样的校验也可以用来校验for循环电路，由于for循环每次运行复杂度不同，可以把每次for循环的验证电路和vk输入到上面的验证电路中，由于验证电路是固定的所以可以直接把验证电路的vk存到以太坊一层中实现验证。

# UltraPlonk
[参考](https://drive.google.com/file/d/1PmAnetV2yVuO94Z68Wnc9QKHw07fBaTo/view?usp=sharing)

UltraPlonk对电路进行了优化，增加了查找表和定制门。承诺部分并没有区别。

## 查找表
**问题**：对于以太坊中的哈希运算中大量布尔运算和循环运算需要大量Plonk门实现。对于这些应用比较多的运算可以把输入输出表格公开给出。使用累加器证明保密数据在表格中，就能证明输入输出是正确的。

### 多项式和集合划分
对于整数n，d，向量 $f\in \mathbb F^n$，$t\in \mathbb F^d$。使用$f \subset t$代表f在t中。构造一个辅助向量$s\in \mathbb F^{n+d}$，重排f与t按照t的顺序把相同的f插入t。生成随机数$\beta\gamma$构造多项式：
$$
F(\beta,\gamma) = (1 + \beta)^n\times \Pi(\gamma + f_i)\times \Pi(\gamma(1+\beta) + t_i + \beta \times t_{i+1})
$$
$$
G(\beta,\gamma) = \Pi (\gamma(1+\beta) + s_i + \beta\times s_{i+1})
$$
可见s构造的式子在$s_i = s_{i+1}$的时候退化成f构造的部分。于是两个式子相等当且仅当$f\subset t$

## Plookup
对于预计算的table多项式$t\in \mathbb F_{<n+1}[X]$和证明方的保密多项式$f\in \mathbb F_{<n}[X]$，witness由多项式表达。
* 零$s\in\mathbb F^{2n+1}$为多项式，使用两个多项式$h_1,h_2\in \mathbb F_{<n+1}[X]$来表达s（因为可能多项式长度会过大）
$$
h_1(g^i) = s_i, \quad i \in [n+1]
$$
$$
h_2(g^i) = s_{i+n}, \quad i \in [n+1]
$$
* 证明方生成多项式承诺，使用哈希生成随机数$\beta\gamma$
* 计算多项式$Z\in\mathbb F_{<n+1}[X]$：
$$
Z(g^i) = \frac{(1+\beta)^{i-1} \times \Pi(\gamma + f_i) \times \Pi(\gamma(1+\beta) + t_j + \beta \times t_{j+1})}{\Pi(\gamma(1+\beta) + s_j + \beta\times s_{j+1})(\gamma(1+\beta) + s_{})}

