# zk_snark学习
[TOC]

# 什么是零知识
本笔记主要针对zk_snark算法以及零知识证明。zk-SNARK 是 Zero-knowledge succinct non-interactive arguments of
knowledge，为了解决证明某个声明为真但又不透露任何秘密信息的知识的算法。
## 要素
零知识证明的系统中有一个验证者（verifier）和一个证明者（prover）。verifier知道prover的一个声明是正确的，但又不知道任何隐私信息。过程中的三个性质：
* 完整性：只要陈述是正确的，证明者就可以让验证者确信。
* 可靠性——如果陈述是错误的，那么作弊的证明者就没有办法让验证者相信。
* 零知识——协议的交互仅仅揭露陈述是否正确而不泄漏任何其它的信息。
## 相关算法
zk_SNARKs, zk_STARKs, BulletProofs是主要应用于区块链的零知识证明算法。他们的性能关系：
[img](./imgs/zk_snark1.png)
