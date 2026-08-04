# 无监督跨相机牛只重识别：消融研究与标签效率分析
# Unsupervised Cross-Camera Cattle Re-Identification: Ablation Study and Label-Efficiency Analysis

> 版本 v1 / Version v1 · 数据集 `2025Sep18`（7 路相机，98 头牛，约 2.2 小时单场次，约 12.4 万张裁剪图 / 7 cameras, 98 identities, one ~2.2 h session, ~124 k crops）
>
> **缩写与全称对照见文末术语表。首次出现均给出全称。/ All abbreviations are spelled out on first use; see the glossary at the end.**

---

## 1. 实验设置 / Experimental Setup

**协议 / Protocol.** 留一相机跨视角评测（leave-one-camera-out）：以斜视相机 **66.130** 的轨迹片段（tracklet）为查询集（query），其余 6 路相机为图库集（gallery），采用 Market-1501 的跨相机检索规则（同一身份且同一相机的样本作为 junk 移除，命中必须来自**不同相机**）。这是本数据最难的斜视-俯视跨视角场景。

**指标 / Metrics.**
- **Rank-1 / Rank-5（首位命中率 / 前五命中率，来自累积匹配特征曲线 Cumulative Matching Characteristics, CMC）**
- **mAP（平均精度均值 / mean Average Precision）**

**骨干网络 / Backbone.** DINOv2（self-**DI**stillation with **NO** labels, version 2，一种无标签自监督预训练方法）的 **ViT-B/14（Vision Transformer, Base 变体, patch 14）**，部分微调最后 4 个 Transformer 块 + 最终归一化层（无低秩适配 Low-Rank Adaptation, LoRA），自动混合精度（Automatic Mixed Precision, AMP）训练。

**训练信号（全程无身份标签）/ Training signal (identity-label-free throughout).** 相机内伪标签（单相机多目标跟踪器给出的轨迹片段）+ 相机间聚类（相机感知 Jaccard 距离 Camera-Aware Jaccard, CA-Jaccard + 基于密度的噪声应用空间聚类 Density-Based Spatial Clustering of Applications with Noise, DBSCAN）+ 跨视角最优传输（Optimal Transport, OT）必连边挖掘 + 相机拓扑不可连接（cannot-link）约束 + 簇对比（Cluster-Contrast）。真值标签仅用于评测打分。

---

## 2. 主消融表 / Main Ablation Table

同一 ViT-B checkpoint，仅改变**检索期读出与后处理**（零重训、零标签），留一相机 66.130：

| # | 方法 / Method | Rank-1 | Rank-5 | mAP | 说明 / Note |
|---|---|---|---|---|---|
| 0 | 256 维投影头 + 余弦（旧协议）/ 256-d projection head + cosine | 0.503 | 0.736 | 0.330 | 此前所有报告数字的读出层 |
| 1 | 768 维骨干池化特征 + 余弦 / 768-d backbone feature + cosine | **0.620** | **0.841** | 0.347 | 批归一化颈思路：检索用投影**前**特征 |
| 2 | + 逐相机中心化 / + Camera-wise Centering (CC) | 0.656 | 0.834 | 0.378 | 减去每相机均值再归一化 |
| 3 | + 相机感知 Jaccard 重排序 / + CA-Jaccard Re-Ranking (RR) | 0.663 | 0.791 | 0.408 | k1=20, k2=6 |
| 4 | **+ 倒数排名融合 / + Reciprocal Rank Fusion (RRF)** | **0.706** | **0.859** | **0.423** | 融合 CC、主成分白化、CC+RR 三视角，k=20 |
| 5 | 4 + 时空掩码 / 4 + Spatio-Temporal (ST) mask | 0.699 | 0.865 | 0.425 | 安全但增益微弱（见 §3.5） |

**主结论 / Headline:** 纯无监督（label-free）rank-1 从 **0.503 提升到 0.706（+20.3 个百分点）**，rank-5 达 0.859，mAP 达 0.423，**全程零重训、零标签**——提升几乎全部来自检索协议修正与后处理杠杆，而非新的表征训练。

对照的**全监督上限**（同骨干、同评测）：768 维特征 + 余弦 + 重排序 = **rank-1 0.963 / rank-5 0.969 / mAP 0.920**。

---

## 3. 各推理期杠杆分析 / Analysis of Inference-Time Levers

### 3.1 从投影头切换到骨干特征（最大单一杠杆）/ Projection head → backbone feature (biggest single lever)
此前检索用的是 256 维投影头输出。改用投影**前**的 768 维骨干池化特征（对应批归一化颈 Batch Normalization Neck, BNNeck 的标准做法：度量/检索用投影前特征），在**同一 checkpoint** 上使无监督 rank-1 由 0.503 升至 0.620、有监督由 0.853 升至 0.908。这是**配对受控**结果（同 checkpoint，仅换读出层），故可靠、非种子噪声。

### 3.2 逐相机中心化 / Camera-wise Centering (CC)
减去每相机特征均值再 L2 归一化，直接抵消斜视-俯视相机偏置。**帮无监督（+3.6 rank-1）但伤有监督**——对已强判别的监督特征属于过度校正。

### 3.3 主成分分析白化 / Principal Component Analysis Whitening (PCAW)
在图库集上拟合白化变换（无标签）。牛皮特征高度相关，余弦相似度被少数全局光照方向主导；白化拉平各维，rank-1 → 0.681、rank-5 → 0.865，但拉散长尾使 mAP 下降——**提升首位命中、牺牲平均精度**。

### 3.4 倒数排名融合 / Reciprocal Rank Fusion (RRF)
不信任任何单一距离，融合三个互补视角（CC 余弦、PCAW 余弦、CC+重排序）的**排名**。融合后 rank-1 达 **0.706**，同时保住 mAP 0.423——各视角的失败模式不同，融合起到互补去噪作用。这是本无监督配方的最佳组合。

### 3.5 时空不可能性掩码 / Spatio-Temporal (ST) impossibility mask
物理约束：非重叠相机对 + 时间区间相交 ⇒ 同一时刻不可能是同一头牛，置无穷大距离。**安全**（仅误杀 0.37% 真匹配，0 个查询变空），但在单相机留出协议上**增益微弱**（每查询仅剪除约 2.7% 图库，mAP +0.001~0.003），因单场次内可利用的时间共现有限；预期在全拓扑部署中价值更大。

### 3.6 无效杠杆（诚实记录）/ Ineffective levers (recorded for honesty)
- **α 查询扩展 / Alpha Query Expansion (AQE)**：两种特征上均退步——单色相似奶牛污染扩展。
- **数据库端增强 / DataBase-side Augmentation (DBA)**：退步（0.66→0.63），同上成因。
- **256 维 + 768 维集成 / emb256 ⊕ feat768 ensemble**：中性，256 维本身更弱。

---

## 4. 两次训练期攻坚——均为负面结果 / Two Training-Time Attacks — Both Negative

我们针对真瓶颈（跨相机关联 / 伪标签质量）做了两次训练期改进，**均未能超过 0.706，此为诚实记录的负面结果，对论文的"我们尝试过什么"叙述至关重要。**

### 4.1 广义均值池化 + 批归一化颈 / Generalized Mean pooling (GeM) + BNNeck
**动机 / Motivation.** 用对 DINOv2 patch token 的空间 GeM 池化 + 时序 GeM + BNNeck 头，替换 CLS（class token 分类标记）/ 注意力池化 / 256 维投影栈，理论上是更适合检索的描述子。**全程无监督**，训练流程不变，1000 步。

**结果 / Result.** 退步：GeM 余弦 rank-1 **0.429**，最佳融合仅 **rank-1 0.528 / rank-5 0.767 / mAP 0.337**（对比 0.706 / 0.859 / 0.423）。

**机理 / Mechanism.** (1) patch token GeM 需先截断到非负（clamp）才能做分数次幂，丢弃了 DINOv2 token 有意义的**负分量**；对单色奶牛，GeM 放大峰值/背景激活而非微弱的毛色纹理，而 CLS token 是 DINOv2 专门训练的整体语义描述子，本就更强。(2) 可学习的幂指数 p 几乎未动（截断+幂的梯度过弱）。(3) 全新初始化的 BNNeck 在 1000 步无监督伪标签下欠训练。

### 4.2 相机中心化空间的自举伪标签 / Bootstrapping pseudo-labels in the CC space
**动机 / Motivation.** 既然 CC 提升**检索**，则在 CC 空间做聚类与跨视角挖掘，应能提高伪标签精度、更好地跨相机合并同一头牛。从最佳 checkpoint 热启动，聚类与最优传输挖掘前均先做 CC，续训 1000→1600 步。

**结果 / Result.** 退步：最佳融合 **rank-1 0.564 / rank-5 0.773 / mAP 0.355**。

**机理 / Mechanism.** CC 空间挖出的必连边**翻倍**（147→208~245 条），聚类数降到 71~77（貌似接近真值 81），**但精度未提高，仍为约 0.12~0.15**——于是 2 倍的边 × 同样 15% 精度 = 2 倍的**错误**跨相机合并，把不同的牛并入同簇，特征退化。**关键洞见：CC 是好的"检索期"操作（每相机全局平移有助排序），却是坏的"聚类空间"**——去除每相机均值会抹掉让单色相似牛彼此区分的判别信号，最优传输遂自信匹配错误裁剪图；接近真值的簇数是**过度合并造成的假象**，成分是错的。

### 4.3 两次负面结果的共同结论 / Joint conclusion
连同此前的多种子确认，**纯无监督在本数据上的平台期为 rank-1 ≈ 0.706 / rank-5 ≈ 0.86 / mAP ≈ 0.42，是数据本身的限制**——跨视角最优传输挖掘精度被稀疏相机重叠 + 单色牛皮钉在约 15%，无法通过网络结构或特征空间技巧在无标签条件下突破。

---

## 5. 标签效率——破局之路 / Label Efficiency — The Way Through

既然纯无监督受限于跨相机关联噪声，唯一被验证能突破的路径是**少量登记标签**（半监督：入栏时登记一部分身份的真值，其余聚类）。标签效率曲线（留一相机 66.130）：

| 登记比例 / Enrolled fraction | Rank-1 | Rank-5 | mAP |
|---|---|---|---|
| 0%（= 无监督平台）| 0.476 | 0.715 | 0.305 |
| 10% | 0.559 | 0.767 | 0.375 |
| 25% | 0.694 | 0.853 | 0.546 |
| **50%** | **0.835** | **0.932** | 0.700 |
| 100%（= 全监督上限）| 0.850 | 0.924 | 0.822 |

**结论 / Conclusion.** **登记约 50% 的牛群 → rank-1 0.835 / rank-5 0.932，基本追平全监督**；25% 已使 rank-5 达 0.853。mAP 需更多标签方能饱和。

> **诚实性说明 / Honesty note.** 本标签效率曲线来自较早的 **ViT-S（ViT-Small）**微调实验（2 个随机种子），与 §2–4 的 ViT-B 结果骨干不同，端点数值不可直接与 ViT-B 的 0.706/0.963 对齐；定性结论（"约 50% 登记 ≈ 全监督"）稳健，但相机就绪版（camera-ready）应在 ViT-B 上以 3~5 个种子重跑该曲线。

---

## 6. 总体结论 / Overall Conclusion

1. **纯无监督的最大杠杆是检索协议本身**（骨干特征 + 白化 + 排名融合），零重训将 rank-1 由 0.503 推至 **0.706**。
2. **表征/结构并非无监督瓶颈**——两次训练期攻坚（GeM+BNNeck、CC 自举）均退步，共同印证瓶颈是**跨相机关联精度**，且为**数据限制**（稀疏重叠 + 单色外观）。
3. **突破 0.706 的唯一验证路径是少量监督**：约 50% 入栏登记的半监督即可追平全监督（rank-1 0.835）。
4. **全监督上限**（ViT-B + 768 维特征 + 重排序）= rank-1 0.963 / rank-5 0.969 / mAP 0.920。

---

## 术语表 / Glossary of Abbreviations

| 缩写 | 全称（英） | 全称（中） |
|---|---|---|
| Re-ID | Re-Identification | 重识别 |
| SSL | Self-Supervised Learning | 自监督学习 |
| DINOv2 | self-DIstillation with NO labels, v2 | 无标签自蒸馏（第 2 版） |
| ViT / ViT-S / ViT-B | Vision Transformer / Small / Base | 视觉 Transformer / 小 / 基础 |
| CLS token | Classification token | 分类标记 |
| GeM | Generalized Mean (pooling) | 广义均值池化 |
| BNNeck | Batch Normalization Neck | 批归一化颈 |
| CC | Camera-wise Centering | 逐相机中心化 |
| PCA / PCAW | Principal Component Analysis / PCA Whitening | 主成分分析 / 主成分白化 |
| RR | Re-Ranking | 重排序 |
| CA-Jaccard | Camera-Aware Jaccard (distance) | 相机感知 Jaccard 距离 |
| RRF | Reciprocal Rank Fusion | 倒数排名融合 |
| ST | Spatio-Temporal | 时空 |
| OT | Optimal Transport | 最优传输 |
| DBSCAN | Density-Based Spatial Clustering of Applications with Noise | 基于密度的噪声应用空间聚类 |
| IICS | Intra-Inter Camera Similarity | 相机内-相机间相似度 |
| CMC | Cumulative Matching Characteristics | 累积匹配特征曲线 |
| mAP | mean Average Precision | 平均精度均值 |
| Rank-1 / Rank-5 | Rank-1 / Rank-5 accuracy | 首位命中率 / 前五命中率 |
| AQE | Alpha Query Expansion | α 查询扩展 |
| DBA | DataBase-side Augmentation | 数据库端增强 |
| TTA | Test-Time Augmentation | 测试时增强 |
| AMP | Automatic Mixed Precision | 自动混合精度 |
| LoRA | Low-Rank Adaptation | 低秩适配 |
| AdamW | Adam with decoupled Weight decay | 解耦权重衰减的 Adam 优化器 |
| tracklet | — | 轨迹片段 |
| query / gallery | — | 查询集 / 图库集 |
| cannot-link | — | 不可连接（约束） |

---

## 相关文件 / Related Files
- 主推理杠杆脚本 / Inference-lever scripts: `new_levers.py`, `new_levers2.py`, `st_validate2.py`, `cowreid/st_inference.py`
- 训练期负面结果 / Training-time negatives: `vitb_unsup_gembn.py`（GeM+BNNeck）, `vitb_unsup_boot.py`（CC 自举）
- 标签效率 / Label efficiency: `label_efficiency.py`
- 结果存档 / Result JSON: `artifacts2/st_final_comparison_v1.json`, `artifacts2/new_levers2_v1.json`, `artifacts2/gembn_eval_v1.json`, `artifacts2/boot_eval_v1.json`
