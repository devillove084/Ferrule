# 实测

2×RTX3090/sm86/driver575.57.08，固定CUDA12.4.99/NCCL2.27.5。6个fresh process（A/B/C+计数），各5轮。

每卡NVML MiB：

| 组 | 5轮drop后 | owner drop | free |
|---|---|---:|---:|
| A：CUDA同模块 | 267×5 | 265 | 23860 |
| B：另dlopen，无comm | 267×5 | 265 | 23860 |
| C：init/warm/gather/comm drop | 381×5 | 379 | 23746 |

起始used0/free24125。第1→5轮CU free增长各为**0 bytes**；输入期间allocator reserved每卡32768，释放后0。

**A已证明265 MiB常驻不是NCCL才有。C首次追加114 MiB，5轮无逐comm增长；非普遍无泄漏证明或硬上界。**

flags=0；primary active从0→1，owner drop后仍1。GetState无refcount，Rust weak消失≠runtime全局owner消失。

独立CUPTI12.4.99另占12 MiB/卡；成功API合计：

| 计数 | A/B | C |
|---|---:|---:|
| primary retain/release | 4/2 | 6/4 |
| 上下文创建/销毁开始 | 2/0 | 2/0 |
| 流创建/销毁 | 6/6 | 36/36 |
| 普通分配/释放 | 14/14 | 14/24 |
| VMM创建/释放 | 0/0 | 81/161 |
| map/unmap | 0/0 | 80/80 |

core追加2次retain；C额外2次于库drop释放，最终差额同A，**并非refcount**。句柄导入等未配对，次数非净分配。C首轮8次非成功回调未细分；操作成功，后续无新增。

scope：session释放comm/backing/streams，runtime/primary/module按process处理。private bound仍Unknown，无hardcap/public。

退出ECHILD、八卡idle、双锁释放。无reset、额外release或killforeign。

证据：run `ownership-nccl2275-cuda124-20261011-01`的summary/policy/events/SHA。CUPTI以SHA/dladdr核对；旧maps未含cupti。

安全下一步：按process-scope补overflow、2site、独立进程Unknown；历史失败不改通过。
