# 基于bert4keras和mt5模型的cache实现  
cacheT5.py是其底层实现  
demo.py是一个简单的演示demo  
https://zhuanlan.zhihu.com/p/75796168 关于cache技术的介绍  
总体来说cache是一种无损的提速方法，这个仓库主要开源我通过bert4keras实现的mt5版cache  

# 性能演示  
测试环境是1660gpu和i5-9400Fcpu  
cpu环境下512-512的输入输出  
整体性能提升了四倍左右  
![R@Q_IV}H8MJA584_`KG1EPJ](https://user-images.githubusercontent.com/62837036/178694163-fce79628-1984-40ae-b2d8-d142cdf6e817.png)  
cpu环境下768-768的输入输出

![image](https://user-images.githubusercontent.com/62837036/178699319-d5849130-76a1-45f8-858d-4ae68701b43f.png)  
![image](https://user-images.githubusercontent.com/62837036/178699354-24918108-f9cc-4ff3-87d2-87a133ec47b0.png)  
提速在九倍左右  
gpu环境下512-512的输入输出    
整体速度提升了两倍左右  
![@CA}91O7{3E))LY2496D3D2](https://user-images.githubusercontent.com/62837036/178694020-520acc44-46d5-4fa4-b083-c35d90ae7b08.png)  

gpu环境下768-768的输入输出     
![XJMA5A2~{2 IDR_JB9)Y(2M](https://user-images.githubusercontent.com/62837036/178694301-f65f4810-2ad4-4cf5-af4d-e538a8c14de9.png)  
不难发现更长的输入输出能让加速比有所提升。   
另外gpu和cpu的提速比差异较大，个人怀疑是并行能力导致的。不过笔者硬件知识堪忧，希望大佬指教  
