# Forward Time Metrics 使用与测试说明

## 1. 功能概述

在模型 forward 调用边界记录 NPU Device Event，异步采集设备端 forward 耗时，按 `(rank, role, phase, batch_size)` 聚合，周期性输出统计日志。

核心特点：
- **非阻塞**：不调用 `synchronize()`/`wait_event()`，不影响调度热路径
- **有界内存**：pending queue 容量 = window_size，事件池上限 2×window_size
- **fail-open**：采集异常不中断推理，不掩盖业务异常
- **覆盖全路径**：target/verify（execute_model 业务路径）、draft（proposer 的 `_runnable` 边界，覆盖 eager 和 ACL graph replay）

## 2. 配置方法

### 2.1 推荐方式：additional-config（主通道）

在 vllm serve 的 `--additional-config` JSON 中配置：

```json
{
  "forward_time_metrics_config": {
    "enabled": true,
    "window_size": 1000,
    "target_batch_sizes": [4, 5, 6]
  }
}
```

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | 是否启用打点 |
| `window_size` | int | `1000` | 每个窗口接受的 forward 数；窗口满后输出一行日志 |
| `target_batch_sizes` | list[int] | `[]`(空=全部) | 只采集指定 BS 的 forward；空列表表示采集所有 BS |

> **为什么用 additional-config 而不是环境变量**：vLLM V1 的 spawn 部署（`VLLM_WORKER_MULTIPROC_METHOD=spawn`）会重建 worker 子进程环境，`export` 的环境变量到不了 worker。additional-config 随 `VllmConfig` 序列化下发，所有 executor 形态（multiproc/ray）都可靠到达。

### 2.2 环境变量 fallback

功能关闭时（`enabled=false`）不会读环境变量，因此残留的错误配置不会阻断启动。仅在 `additional-config` 未配置 `forward_time_metrics_config` 时，以下环境变量作为 fallback（适用于离线推理 `LLM()` 等 fork 模式场景）：

```bash
export VLLM_ASCEND_ENABLE_FORWARD_TIME_METRICS=1
export VLLM_ASCEND_FORWARD_TIME_WINDOW_SIZE=1000
export VLLM_ASCEND_FORWARD_TIME_TARGET_BATCH_SIZES=4,5,6
```

### 2.3 在启动脚本中注入（模板翻译）

如果习惯用环境变量操作，可以在 bash 启动脚本里把 env 翻译进 additional-config（bash 层环境变量是完整的）：

```bash
FTM_ENABLED="${VLLM_ASCEND_ENABLE_FORWARD_TIME_METRICS:-0}"
FTM_WINDOW="${VLLM_ASCEND_FORWARD_TIME_WINDOW_SIZE:-1000}"
ADDITIONAL_CONFIG="{\"forward_time_metrics_config\": {\"enabled\": ${FTM_ENABLED}, \"window_size\": ${FTM_WINDOW}}, \"use_offload\": false, ...}"
```

## 3. 日志格式

### 3.1 启用日志

collector 初始化时输出一行确认：

```text
ForwardTimeCollector enabled: rank=0 window_size=100 target_batch_sizes=all
```

### 3.2 窗口统计日志

每攒满 window_size 次 forward 输出一行（按 `rank/role/phase/bs` 分组）：

```text
Forward timing: window=1 rank=0 role=target phase=decode bs=4 count=238 avg_ms=7.421 min_ms=7.106 max_ms=8.337 dropped=0 errors=0
```

| 字段 | 含义 |
| --- | --- |
| `window` | 进程内单调递增的窗口 ID，从 1 开始 |
| `rank` | NPU worker rank |
| `role` | `target`（主模型）或 `draft`（草稿模型） |
| `phase` | `prefill`、`decode`、`mixed`、`verify` |
| `bs` | 未 padding 的实际请求数 |
| `count` | 本 key 在本窗口内成功计时的 forward 次数 |
| `avg_ms` | 设备 forward 平均耗时 |
| `min_ms` / `max_ms` | 最小 / 最大耗时 |
| `dropped` | 因队列准入失败或 teardown 未完成而丢弃的样本数 |
| `errors` | Event API 或耗时值异常数 |

### 3.3 尾窗日志

进程优雅停机时（`NPUModelRunner.shutdown()`），未满的窗口也会强制输出（`final=True` flush），未完成的 pending 样本计入 `dropped`。

## 4. 如何看打点日志

### 4.1 时间戳注意事项

日志行的时间戳是**窗口吐出时间**，不是 forward 的执行时间。窗口攒满 50-1000 次 forward 才吐一行，实际 forward 比时间戳早一个窗口周期（decode 侧约 6-120 秒，视 window_size 而定）。

### 4.2 与 benchmark 数据交叉验证

打点数据可与 aisbench 等测试工具的端到端指标对账：

| 端到端指标 | 打点数据计算方法 | 验证示例 |
| --- | --- | --- |
| **TPOT** | `(verify_avg + draft_avg) ÷ tokens_per_step` | 122.5ms ÷ 2.98 = 41.1ms ✓ |
| **TTFT** | `input_tokens ÷ prefill_throughput` | 1M ÷ 8929 t/s = 112.0s ✓ |
| **BS 分布** | 从日志的 `bs=` 字段统计 | c32÷8 实例=4，实际峰值 3（1M prefill 耗时长，错峰进场） |

其中 `tokens_per_step = step_time ÷ TPOT`，反推 MTP 接受率：`acceptance = (tokens_per_step - 1) / num_speculative_tokens`。

### 4.3 窗口记账闭合校验

同一窗口同一 rank 的所有 key 的 `count` 之和应等于 `window_size`。例如 MTP spec=3 下，draft 和 verify 以 1:1 交替，各得 window_size/2：

```text
window=416: draft decode bs=1 count=50 + target verify bs=1 count=50 = 100 ✓
```

含首次 decode 步的窗口（target phase=decode，count=1），verify 少 1：

```text
window=544: draft 50 + verify 49 + target_decode 1 = 100 ✓
```

### 4.4 异常判断

- `dropped > 0`：设备积压或 teardown 时有未完成样本，需结合负载评估是否正常
- `errors > 0`：Event API 异常，检查 NPU 驱动状态
- `max_ms >> avg_ms`：有离群 forward（如 ACL Graph 首次捕获、驱逐干扰），查看 max 对应的时间点

## 5. role 和 phase 映射

### 5.1 Target/Verify（主模型）

| 条件 | role | phase |
| --- | --- | --- |
| `DecodeOnly` | target | decode |
| `SpecDecoding` 且 `scheduled_spec_decode_tokens` 非空 | target | verify |
| `SpecDecoding` 且无 scheduled draft tokens | target | decode |
| `PrefillNoCache` / `PrefillCacheHit` | target | prefill |
| `ChunkedPrefill` | target | mixed |

阶段映射使用 **pre-overlay** 的 attention state（`_build_attn_state` 的返回值），因为 non-MTP（EAGLE 系）的 verify 步会把 `self.attn_state` 改写成 `ChunkedPrefill`。同时联合 `scheduler_output.scheduled_spec_decode_tokens` 判定，避免 MTP + PD disagg 下纯 decode 步被误标为 verify。

### 5.2 Draft（草稿模型）

| 条件 | role | phase |
| --- | --- | --- |
| 生成阶段 | draft | decode |
| 明确执行 prefill | draft | prefill |
| 无法可靠分类 | draft | mixed |

draft forward 在 `_runnable` 调用边界埋点（不是 `_run_merged_draft` 内部的 `self.model` 调用），因为 ACL graph replay 不重跑 Python 体，内部调用在图模式下不执行。dummy run / graph capture / profile warmup 不计入业务统计。

### 5.3 主模型 BS

```python
batch_size = self.input_batch.num_reqs  # 未 padding 的实际请求数
```

draft BS 取 `self.runner.input_batch.num_reqs`（与 target 同源），不使用从 token 数推导的 `batch_size`。

## 6. 测试

### 6.1 单元测试

测试位于 `tests/ut/core/`，使用 Fake Event 工厂注入，CPU 环境可直接运行（无需 NPU）：

```bash
cd /workspace/vllm-ascend-dev
pytest tests/ut/core/test_forward_time_collector.py -v
pytest tests/ut/core/test_forward_time_model_runner.py -v
pytest tests/ut/core/test_forward_time_proposer.py -v
```

覆盖场景（95 项）：

- **collector 语义**：关闭路径、正常计时、非阻塞保证（无 synchronize/wait_event）、窗口边界、多 BS 分组、角色阶段分组、BS 过滤、队列上限、未完成样本、异常路径、非法耗时、配置校验、forward 异常、abort 计入 dropped、事件池复用与上限、日志排序、FIFO 队首停止、flush 幂等
- **model runner 接入**：decode/verify/prefill/mixed 阶段映射、MTP+PD disagg 纯 decode 不误标 verify、BS 取 input_batch.num_reqs、_dummy_run 不计入统计
- **proposer 接入**：draft BS 取真实 num_reqs、eagle/Step3.5/Medusa/ExtractHiddenStates 各路径覆盖

### 6.2 集群验证

在真实 PD 分离集群上的验证步骤：

1. 四台机器的 ascend-zzz 容器同步代码到 `feat/forward-time-metrics` 分支
2. 在 P 机和 D 机的启动脚本 `--additional-config` 中注入打点配置
3. 按序拉起：memcache → decode → prefill → proxy
4. 确认每个 worker 输出 `ForwardTimeCollector enabled` 启动日志
5. 跑流量，收集 `Forward timing:` 日志行
6. 交叉验证：`dropped=0 errors=0`、BS 分布与并发匹配、TPOT/TTFT 对账

验证参考数据（GLM-5.2, 4 节点 PD, 1M×c32×pc0.99）：

```text
window=1 rank=2 role=target phase=verify bs=1 count=50 avg_ms=92.3 min_ms=75.3 max_ms=102.0 dropped=0 errors=0
window=1 rank=8 role=draft  phase=decode bs=1 count=50 avg_ms=30.2 min_ms=27.0 max_ms=43.3 dropped=0 errors=0
```

TPOT 对账：`(92.3 + 30.2) ÷ 2.98 = 41.1ms`（与 aisbench TPOT 精确匹配，反推 MTP spec=3 接受率 66%）。

### 6.3 性能验收

A/B 测试（相同模型、并行配置、输入数据、并发、预热轮数和测量时长）：

- Baseline：打点关闭
- Test：打点开启，window_size=1000，采集全部 BS

验收条件：
- 稳态吞吐下降 ≤ 3%
- 无新增 CPU 等待或 NPU 全局同步
- pending queue 峰值不超过 window_size
- dropped 和 errors 接近 0

## 7. 已知限制

- **Medusa draft**：业务路径的 draft forward 在上游 vllm 代码中（`vllm/v1/spec_decode/medusa.py`），通过 `run_timed_draft_forward` 模块级封装覆盖，但封装在 proposer 基类侧，非 medusa 专属逻辑
- **窗口时间戳**：日志时间戳是窗口吐出时间，非 forward 执行时间；精确关联需加 start/end 时间字段（设计文档已留扩展点）
- **多 stream**：首版假设同一 collector 的样本在同一执行 stream 上有序提交；多 stream 需按 stream 拆分 pending queue
- **ACL Graph 懒捕获**：新 batch descriptor 首跑走 capture 分支，首个样本的 max_ms 会带秒级离群值（真实设备时间，非噪声）
- **Pipeline Parallel**：各 rank forward 耗时可能不同，首版保留每 rank 独立观测，不定义跨 stage 聚合
