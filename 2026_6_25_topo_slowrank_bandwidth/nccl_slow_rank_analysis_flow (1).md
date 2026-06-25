# NCCL Slow Rank Analysis Flow

这个文件描述 `comm-topology-lab` 的完整分析流程。目标是把一份简化 NCCL trace 和一份 GPU 拓扑 JSON 合起来，自动输出：

1. 慢 rank 是谁。
2. 哪条链路最可疑。
3. 为什么怀疑它。
4. 下一步应该用哪些命令验证。
5. 生成可视化图：GPU 拓扑图、rank 延迟柱状图、链路健康热力图。

---

## 1. 输入数据

项目使用两个输入文件：

```text
 data/nccl_trace.log
 data/topology.json
```

### 1.1 `nccl_trace.log`

每一行代表一次简化后的 NCCL 通信事件：

```text
ts_ms op comm rank peer bytes duration_ms link
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `ts_ms` | 事件时间戳，单位 ms |
| `op` | collective 类型，例如 `all_reduce`、`all_gather`、`reduce_scatter` |
| `comm` | NCCL 使用的通信结构，例如 `ring` 或 `tree` |
| `rank` | 当前 rank |
| `peer` | 当前 rank 通信的对端 rank |
| `bytes` | 本次通信数据量 |
| `duration_ms` | 本次通信耗时，越大越慢 |
| `link` | 本次通信经过的逻辑链路，例如 `NVL0`、`PHB0`、`PIX0` |

### 1.2 `topology.json`

拓扑文件描述：

- rank 到 GPU 的映射。
- GPU 所在 NUMA 节点。
- GPU 之间的链路类型。
- 链路理论带宽。
- 拓扑健康状态。

示例链路：

```json
{
  "src": 3,
  "dst": 4,
  "name": "PHB0",
  "type": "PCIe host bridge",
  "expected_gbps": 32,
  "health": "suspect"
}
```

这表示 rank 3 和 rank 4 之间经过一条 PCIe host bridge 路径，并且拓扑层面已经标记为可疑。

---

## 2. 解析 NCCL trace

脚本首先逐行读取 `data/nccl_trace.log`，用正则表达式提取：

```text
rank, peer, bytes, duration_ms, link
```

然后把每一条通信记录转换成 `Event` 对象。

每个 `Event` 会额外计算吞吐：

```text
gbps = bytes * 8 / duration_seconds / 1e9
```

这个值用于判断某条链路是否明显低于预期带宽。

---

## 3. 计算全局基线

脚本会先计算所有通信事件的全局中位数延迟：

```text
global_median_ms = median(all duration_ms)
```

在这个 demo 中，全局中位数大约是：

```text
13.1 ms
```

然后定义慢事件阈值：

```text
slow_threshold = global_median_ms * slow_factor
```

默认：

```text
slow_factor = 1.8
```

所以只要某次通信耗时明显高于全局中位数，就会被标记为 slow event。

---

## 4. Rank 维度分析

脚本会按 rank 分组，统计每个 rank 的：

| 指标 | 含义 |
| --- | --- |
| `median_ms` | 该 rank 的中位数延迟，反映整体是否偏慢 |
| `max_ms` | 该 rank 的最大延迟，反映最坏情况 |
| `min_gbps` | 该 rank 的最低吞吐，反映是否出现带宽塌陷 |
| `slow_events` | 该 rank 出现慢事件的次数 |
| `links` | 该 rank 经过过哪些链路 |

判断慢 rank 时，不只看单次最大值，而是综合：

```text
rank_score = median 相对全局中位数的涨幅 + slow_events 惩罚项
```

这样可以避免一个 rank 只因为偶发尖刺就被误判。

在当前样例中：

```text
rank 3 多次经过慢路径，并且 median_ms 明显高于其他 rank
```

所以报告给出：

```text
Most likely slow rank: rank 3
```

---

## 5. Link 维度分析

脚本还会按 link 分组，统计每条链路的：

| 指标 | 含义 |
| --- | --- |
| `median_ms` | 经过该链路时的中位延迟 |
| `max_ms` | 经过该链路时的最大延迟 |
| `avg_gbps` | 平均吞吐 |
| `expected_gbps` | topology JSON 中记录的理论带宽 |
| `throughput_ratio` | 实测吞吐 / 理论带宽 |
| `slow_events` | 该链路关联的慢事件次数 |
| `health` | topology JSON 中的健康状态 |

一条链路会被认为可疑，如果满足任意条件：

```text
slow_events > 0
或 health == suspect
或 throughput_ratio < 0.55
```

在当前 demo 中，最可疑的是：

```text
PHB0
```

原因是：

1. `PHB0` 关联了多个 slow event。
2. `PHB0` 是跨 GPU3 和 GPU4 的路径。
3. 该路径类型是 `PCIe host bridge`。
4. topology JSON 已经把它标记为 `suspect`。
5. note 中提示：`cross-socket path; prior DCGM saw replay counter bumps`。

因此初步判断不是 NCCL 算法本身的问题，而更像是：

```text
PCIe / NUMA / host bridge / 拓扑映射 / 硬件链路健康问题
```

---

## 6. 可视化输出

脚本会自动生成三张图：

```text
reports/images/topology_slow_link.png
reports/images/rank_latency_bar.png
reports/images/link_health_heatmap.png
```

### 6.1 GPU 拓扑图

拓扑图展示 GPU 之间的连接关系：

- 每个节点代表一个 rank/GPU。
- 边代表 topology JSON 中的链路。
- 慢链路或可疑链路会被高亮。
- `PHB0` 会用红色高亮，帮助快速定位异常路径。

这张图用于回答：

```text
慢 rank 在拓扑上连接到了哪里？
慢链路是否跨 NUMA / 跨 socket？
```

### 6.2 Rank 延迟柱状图

柱状图展示每个 rank 的 median latency 和 max latency。

这张图用于回答：

```text
哪个 rank 的延迟整体更高？
哪个 rank 出现了最大尖刺？
```

在当前 demo 中，rank 3 会明显高于其他 rank。

### 6.3 Link 健康热力图

热力图把每条链路的风险拆成几个维度：

| 维度 | 含义 |
| --- | --- |
| `latency_ratio` | 链路中位延迟相对全局中位数的比例 |
| `slow_events` | 慢事件数量归一化 |
| `throughput_deficit` | 实测吞吐低于预期的程度 |
| `topology_flag` | topology 是否标记 suspect |

这张图用于回答：

```text
哪条链路的综合风险最高？
是延迟问题、吞吐问题，还是拓扑健康标记问题？
```

---

## 7. 报告生成

脚本最终生成：

```text
reports/slow_rank_report.md
```

报告包含：

1. 结论。
2. 三张可视化图。
3. Rank summary 表格。
4. Suspicious links 表格。
5. 下一步验证命令。
6. 建议排查顺序。

---

## 8. 下一步验证命令含义

### 查看 GPU 拓扑

```bash
nvidia-smi topo -m
```

用于验证真实机器上的 GPU 拓扑关系，例如：

- GPU 是否经过 NVLink。
- GPU 是否经过同一个 PCIe switch。
- GPU 是否跨 CPU socket。
- GPU 和 NIC 是否靠近。

### 查看 NVLink 状态

```bash
nvidia-smi nvlink --status
```

用于确认 NVLink 是否正常，例如是否出现 link down。

### 监控 GPU 运行状态

```bash
nvidia-smi dmon -s pucvmet -d 1
```

用于观察 GPU 利用率、功耗、温度、显存、PCIe 传输等指标。

### 运行 DCGM 诊断

```bash
dcgmi diag -r 3
```

用于检查 GPU 健康状况。

### 监控 DCGM 指标

```bash
dcgmi dmon -e 1002,1003,1004,1005
```

用于观察 PCIe / NVLink / replay counter 等可能影响通信性能的指标。

### 打开 NCCL Debug

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL ./your_nccl_test
```

用于确认 NCCL 实际选择了什么 ring/tree 图，以及慢路径是否真的在 NCCL graph 里。

### Dump NCCL Topology

```bash
NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml NCCL_DEBUG=INFO ./your_nccl_test
```

用于导出 NCCL 看到的拓扑，和 `nvidia-smi topo -m`、`topology.json` 进行对比。

### 检查 NUMA 和 PCIe 树

```bash
numactl --hardware && lspci -tv
```

用于确认 GPU、CPU socket、PCIe switch、root complex 的真实挂载关系。

---

## 9. 推荐排查顺序

真实排障时建议按这个顺序：

1. 先确认慢 rank：看 rank latency 和 slow events。
2. 再确认慢链路：看 link summary 和 heatmap。
3. 看 NCCL 是否真的选择了这条链路。
4. 对比 `nvidia-smi topo -m` 和 NCCL topo dump。
5. 监控 PCIe / NVLink / DCGM counters。
6. 尝试 rank remapping，避开可疑链路。
7. 如果 remap 后性能恢复，说明拓扑路径强相关。
8. 如果始终只有同一条物理路径慢，继续排查硬件：线缆、riser、switch、host bridge、NUMA placement。

---

## 10. 当前 demo 的结论

当前样例的结论是：

```text
慢 rank: rank 3
可疑链路: PHB0
初步原因: 跨 socket / PCIe host bridge 路径异常，可能伴随 PCIe replay counter 增加
```

更像是：

```text
拓扑/链路/硬件健康问题
```

而不是：

```text
单纯 NCCL 算法问题
```



## 11 colab 输入

 raw version



!unzip -o comm-topology-lab.zip
%cd comm-topology-lab
!python src/find_slow_rank.py
!echo "================ REPORT ================"
!cat reports/slow_rank_report.md



v1

!rm -rf comm-topology-lab
!unzip -o comm-topology-lab-v2.zip
%cd comm-topology-lab
!python src/find_slow_rank.py

from IPython.display import Markdown, Image, display

display(Markdown(open("reports/analysis_flow.md").read()))
display(Markdown(open("reports/slow_rank_report.md").read()))

display(Image("reports/images/topology_slow_link.png"))
display(Image("reports/images/rank_latency_bar.png"))
display(Image("reports/images/link_health_heatmap.png"))
