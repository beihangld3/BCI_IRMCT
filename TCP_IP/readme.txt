这里康复机器人提供server，运行电机控制程序的笔记本称作PC1，首先命令行，ipconfig看一下IP，修改一下host（本机IP address），port。

脑电识别的机器充当客户端，笔记本称为PC2，首先修改自己的IP，跟PC1接近，最后一字节相邻。之后，修改client里面的host和port，和server一样。

先运行server，再运行client，把传送给底层的命令通过client-> server，就实现了同步。



