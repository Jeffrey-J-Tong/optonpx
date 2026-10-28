- [ ] optimize artifact removal in the raw ephys trace
  - [ ] remove milk-drinking artifact
  - [ ] check whether should remove the laser-off artifact
  - [ ] complete notebook 2.2 noise mask version

(2 options: [1] detect based on threshold [2] detect based on behavior signal

## Optotagging 判定：待定 / 待补充事项

### 1. reliability 阈值：3%（论文 / Thomas）vs 25%（Dan）

| 来源                                          | 条件                                                    |
| ------------------------------------------- | ----------------------------------------------------- |
| Faust et al., 2025 (Curr Biol) STAR Methods | 在 10 ms 光照内响应的脉冲 ≥ 3%（任一功率档）                          |
| Thomas 代码（`Fig1wrapper.m` 等）                | `reliability >= 0.03`（先 `round(reliability*100)/100`） |
| Dan pipeline（`opto_1hz_salt.json`）          | `min_reliability = 0.25` 且 `min_responded = 10`       |
| notebook 4.2 现状                             | 同 Dan：reliability ≥ 0.25 且 n_responded ≥ 10           |

- 两个阈值相差约 8 倍，对 tag 数量影响很大：按 3%，100 个脉冲中有 3 次响应即可通过。
- 按 3% 会纳入大量低可靠性的 unit，但只要同时通过 SALT（p < 0.001）和波形相关（r > 0.9），就符合论文的定义。
- 可选做法：两套结果都输出（例如 `response_pass_3pct` 和 `response_pass_25pct`），只用其中一套做最终判定。
- 注意：Thomas 计算 reliability 的代码（`laserResponse3/4.m`）不在 `Faust et al., 2024 scripts` 文件夹里，参数是从预先算好的 `bigUnitparams.mat` 读入的。他的响应窗口和 SALT 窗口细节还没有核实。

### 5. 次要事项（可选）

**5a. Thomas 的两类 " 违规 " 计数（只用于报告和作图，不影响判定）**

- `wfViol`：p < 0.001 且 reliability > 0.02，但波形相关 < 0.9（光诱发波形和 session 平均波形不像）。
- `relViol`：p < 0.001，但 reliability < 0.03（SALT 显著但响应太少）。
- 两类都排除已经被 tag 的 unit（`wfViolIDX(taggedID) = 0` 等）。
- 用于论文 Fig 1E/F：p 值分布中排除这两类 unit；"untagged SPN" 的潜伏期分布也排除这两类。
- 4.2 中可以直接从 `optotagging_summary.csv` 的现有列计算（`salt_p`、`reliability`、`wave_corr`、`is_tagged`），不需要重新跑分析。

**5b. " 未评估 " ≠ 阴性（Dan）**

- Dan 的 `build_single_sort_unit_build.m` 把缺少 opto 指标的 unit 标为 `not_opto_evaluated` / `pMSN_not_opto_evaluated`，不当作 untagged。
- 4.2 会对所有加载的 unit 都做评估，目前不存在这个问题。但如果以后改变 unit 筛选，或者合并多个 session，要注意区分 " 没评估 " 和 " 评估了但没通过 "。

**5c. 潜伏期 / jitter 上限**

- 论文、Thomas、Dan 三者都没有把潜伏期或 jitter 作为判定条件，只报告（论文 Fig 1F 展示首 spike 潜伏期分布）。
- 4.2 已经输出 `latency_ms`、`median_latency_ms`、`jitter_ms`，以后如果需要可以作为额外条件，用来区分直接激活和间接（网络）激活。





关于通过波形对神经元类型进行分类: 问题在于有些位置比较特殊的 neurons 的波形计算波宽有问题
