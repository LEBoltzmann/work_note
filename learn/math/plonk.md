# plonk
本文学习plonk。参考：

[zkSnark-Plonk证明系统](https://drive.google.com/file/d/1-wcIHORaRMkcZX32YBey2boUF5d5Aw1t/view?usp=sharing)

# 密码学承诺
**Schwartz-Zippel引理**：令P为有限域$\mathbb F$上的多项式$P = F(x_1, x_2, 
ldots, x_n)$, 其阶为d。令S为有限域$\mathbb F$的子集，从S中选择随机数$r_1, r_2\ldots, r_n$，则多项式等于零的概率可以忽略，即：

$$
\Pr[P(r_1, r_2, \ldots, r_n) = 0] \leq \frac{d}{|S|}
$$
单变量的情况下，等价于多项式的阶为d，则最多有d个根。

## 
