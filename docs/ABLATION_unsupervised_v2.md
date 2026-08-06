# 无监督跨相机牛只重识别:消融研究、集成蒸馏与协议稳健性
# Unsupervised Cross-Camera Cattle Re-Identification: Ablations, Ensemble Distillation, and Protocol Robustness

> 版本 v2 / Version v2(取代 v1;新增:相机感知代理 CAP、集成蒸馏、多协议验证、部署模式)
> 数据集 `2025Sep18`:7 路相机,98 头牛,单场次约 2.2 小时,约 12.4 万张裁剪图 / 7 cameras, 98 identities, one ~2.2 h session, ~124 k crops
>
> **所有缩写首次出现均给出全称;完整对照见文末术语表。/ All abbreviations are spelled out on first use; see the glossary.**
> **学术图表一律英文**(`figures/*_en_v1.png`);本文档正文保持中英双语。/ All academic figures are English-only; this document stays bilingual.

---

## 1. 实验设置 / Experimental Setup

**基准协议(主协议)/ Primary protocol.** 留一相机跨视角评测(leave-one-camera-out):以斜视相机 **66.130** 的轨迹片段(tracklet)为查询集(query,n=163),其余 6 路相机为图库集(gallery,n=834),采用 Market-1501 跨相机检索规则(同身份同相机为 junk,命中必须来自不同相机)。66.130 的图像**从未参与训练**——这是最严格的"未见相机"设定。

**辅助协议 / Secondary protocols(§6):** 全转导协议(full transductive:查询=图库=全部出现于 ≥2 相机的轨迹片段,943 条)与逐相机查询扫描(per-camera query sweep:7 台相机轮流作查询)。

**指标 / Metrics.** Rank-1 / Rank-5(首位/前五命中率,来自累积匹配特征曲线 Cumulative Matching Characteristics, CMC)与 mAP(平均精度均值 mean Average Precision)。查询集 n=163 时 ±1 个查询 ≈ 0.6 个百分点。

**骨干网络 / Backbone.** DINOv2(self-**DI**stillation with **NO** labels v2,无标签自蒸馏预训练)ViT-B/14(Vision Transformer, Base, patch 14),部分微调最后 4 个 Transformer 块,自动混合精度(Automatic Mixed Precision, AMP)。

**监督界定 / What counts as supervision.** 全程零身份标签;唯一先验是单相机多目标跟踪器输出的轨迹片段(文献标准假设,TAUDL/UTAL)。真值仅用于评测打分与诊断测量。

---

## 2. 主结果 / Main Results

**图 1 / Figure 1:`figures/fig1_journey_en_v1.png`** — 无监督之旅七级台阶。

| # | 方法 / Method | Rank-1 | Rank-5 | mAP |
|---|---|---|---|---|
| 0 | 旧协议:256 维投影头 + 余弦 / old 256-d projection read-out | 0.503 | 0.736 | 0.330 |
| 1 | 768 维骨干池化特征 / 768-d backbone feature(批归一化颈 BNNeck 思路)| 0.620 | 0.841 | 0.347 |
| 2 | + 逐相机中心化 + 重排序 / + Camera-wise Centering (CC) + Re-Ranking (RR) | 0.663 | 0.791 | 0.408 |
| 3 | + 倒数排名融合 / + Reciprocal Rank Fusion (RRF) | 0.706 | 0.859 | 0.423 |
| 4 | + 水平翻转测试期增强 / + horizontal-flip Test-Time Augmentation (TTA) | 0.718 | 0.877 | 0.425 |
| 5 | 相机感知代理 5 种子集成 / Camera-Aware Proxies (CAP), 5-seed ensemble | 0.749 | 0.932 | 0.491 |
| 6 | **集成蒸馏三学生 / ensemble-distilled trio (k=2)** | **0.883** | **0.957** | **0.536** |
| — | 全监督参考 / supervised reference(同骨干 + 重排序)| 0.963 | 0.969 | 0.920 |

**主结论 / Headline.** 纯无监督 rank-1 由 0.503 升至 **0.883**(+38.0 个百分点),rank-5 0.957 已与全监督(0.969)相差 1.2 分。其中 #0→#4 为**零重训**的检索期修正(+21.5),#5→#6 为两级训练期方法(+16.5)。

---

## 3. 检索期杠杆 / Inference-Time Levers(#0→#4)

- **骨干特征读出**(最大单一检索期杠杆,+11.7):此前检索用 256 维投影头输出;改用投影**前**的 768 维池化特征(BNNeck 惯例)。同 checkpoint 配对对照,非种子噪声。
- **逐相机中心化 CC**(+3.6):减每相机均值再 L2 归一化,抵消斜视-俯视偏置。**帮无监督、伤有监督**(对强判别特征属过度校正)。
- **主成分白化 PCAW**(Principal Component Analysis Whitening,图库上拟合):提 rank-1/rank-5、伤 mAP;作为 RRF 的一个视角使用。
- **倒数排名融合 RRF**(+5.0):融合 CC 余弦、PCAW 余弦、CC+重排序三个互补排名(k=20)。
- **翻转 TTA**(+1.2):原图+水平翻转各嵌入一次取平均,rank-1/5/mAP 三指标一致上升。
- **时空不可能性掩码 ST**(Spatio-Temporal mask):非重叠视野相机对 + 时间区间相交 ⇒ 不可能同一头牛。安全(仅误杀 0.37% 真匹配)但功率低(每查询仅剪 ~2.7% 图库),mAP 微增。
- **无效(诚实记录):** α 查询扩展(Alpha Query Expansion, AQE)两处皆退步;数据库端增强(DataBase-side Augmentation, DBA)退步;256⊕768 集成中性。

---

## 4. 制胜方法:相机感知代理 + 集成蒸馏 / The Winning Method: CAP + Ensemble Distillation

### 4.1 第一级:相机感知代理 / Rung 1: Camera-Aware Proxies (CAP)

**动机 / Motivation.** 此前的簇对比(Cluster-Contrast)把一个簇的所有跨相机实例拉向**单一质心**——一条错误合并即污染整簇(挖矿精度仅 ~15%,这是三次训练期失败的共同死因,见 §5)。CAP(参照 CAP/O2CAP, AAAI'21/TIP'22)改为**每(簇,相机)一个代理**:相机内做 softmax 分类;相机间仅把实例**软拉**向同簇异相机代理——错误合并只污染一个代理。

**多种子结果 / Multi-seed(5 个独立种子,冠军配方 + TTA):** 0.804 / 0.472 / 0.724 / 0.669 / 0.687,均值 **0.671 ± 0.110**——单种子方差巨大(1/5 种子聚类中途塌缩,且**无任何无标签信号可识别坏种子**:塌缩种子与正常种子的聚类数轨迹几乎相同;共识加权也失败——最强种子反而最"不合群",剔除它掉到 0.669)。

**种子集成 / Seed ensembling(图 2:`fig2_ensemble_curve_en_v1.png`).** 对全部 C(5,k) 个子集做距离平均集成:k=1→5 均值 0.674→0.742 单调升、标准差 0.108→0 单调缩。**5 种子全集成 = 0.749 / 0.932 / 0.491**(零挑选,可宣称)。

### 4.2 第二级:集成蒸馏 / Rung 2: Ensemble Distillation

**教师信号诊断(真值仅用于测量)/ Teacher-signal diagnostics.**
- 集成空间的**整簇聚类**跨相机对精度仅 ~21%(传递性放大错误)→ 不可用;
- 集成空间的**跨相机互最近邻链接**(mutual k-Nearest-Neighbour links):k=1 共 103 条 @ **60.2%**,k=2 共 287 条 @ 51.9%(对比训练期最优传输挖矿的 15%);
- **关键测量:** 把 60% 精度的链接经并查集**传递合并**成簇后,跨相机对精度塌至 **16.7%** → 链接必须**成对使用、禁止合并**;
- 相机内聚类(集成空间)成对精度 **0.879**(339 簇)——史上最干净的标签。

**学生设计 / Student design.** 相机内簇固定作 CAP 代理 + 每条链接一个**成对**代理拉力(置信度=余弦加权,softmax 分母为全部异相机代理)+ 相机内交叉熵分支;标签全程固定(无刷新→无漂移→无塌缩);从头训 1000 步早停。

**学生结果 / Students.** k=1 学生(2 种子):0.712 / 0.712——方差几乎为零。**k=2 学生(3 种子):0.773 / 0.847 / 0.816,均值 0.812 ± 0.031——单模型即过 0.80**;更高召回(287 条)胜过更高精度(103 条),前提是成对使用把单条错链危害封顶。**三学生距离平均集成 = 0.883 / 0.957 / 0.536。**

**收敛性 / Convergence.** 用变强的 7 模型集成重新挖链接,精度不再提升(0.591 vs 0.602)——蒸馏阶梯**一轮即收敛**;学生复现了教师的链接知识,不产生新信息。

### 4.3 机理小结 / Why this worked where three attacks failed

每一级都对噪声有**结构性容忍**:代理级(而非质心级)损失把错误局部化;成对(而非传递)链接把错误封顶;固定标签+从头训+早停杜绝自训练漂移;种子集成吸收剩余方差。

---

## 5. 负面结果(完整诚实记录)/ Negative Results (all recorded)

**图 5 / Figure 5:`fig5_methods_en_v1.png`** — 12 个方法族得失。

| 方法 / Method | 结果(最佳配置)| Δrank-1 | 失败机理 / Mechanism |
|---|---|---|---|
| GeM 广义均值池化 + BNNeck 换头 | 0.528 / 0.767 / 0.337 | −17.8 | patch token 截断丢负分量;单色牛皮上放大背景激活;可学习幂指数梯度弱;CLS(分类标记)本就是 DINOv2 的强整体描述子 |
| 相机中心化空间自举伪标签 | 0.564 / 0.773 / 0.355 | −14.2 | CC 是好的检索平移、坏的聚类空间:链接翻倍但精度不变(~0.13)→ 2 倍错误合并;"接近真值的簇数"是过度合并假象 |
| DINOv2 patch token 部件匹配(推理期)| 融合后 0.564 | −15.4 | 网格部件在斜视-俯视间不对齐(不似人体部件有语义对齐);部件特征单独仅 0.20–0.26 |
| 续训 / 过训练(严格挖矿对照实验)| 对照 0.626,严格 0.620 | −9.8 | **step 1000 是早停最佳点**;续训 300 步无论松/严挖矿均退步——自训练噪声累积。注:严格组 mAP 0.404 ≫ 对照 0.321 → 链接精度确是真实旋钮,只是不敌过训练 |
| 冻结 ViT-S 多骨干融合 | 0.681 | −2.5 | 弱成员(0.203)拖累强成员 |
| 查询扩展 AQE / 数据库增强 DBA | — | −8.0 / −3.0 | 单色相似牛污染扩展 |
| 种子共识加权 | 0.669(剔除最强种子)| — | 共识=平庸多数;最强种子的独到正确判断被判为异类 |

---

## 6. 协议稳健性验证 / Protocol Robustness(关键节)

**图 3 / Figure 3:`fig3_percamera_en_v1.png`。**

**发现 1:0.883 是协议特定的。** 逐相机查询扫描(无监督三学生):66.130 查询 0.877,**其余 6 台相机仅 0.42–0.56**;全转导 0.516。

**发现 2:不是协议难,是方法偏科。** 同一扫描下**有监督处处均强**(0.877–0.979,全转导 0.956)→ 其他相机的差距是方法性余量,非协议噪声。悖论待解:无监督在**从未见过**的斜视相机上最好、在自己(无标签)训练过的相机上反而弱;有监督无此不对称。

**发现 3:部署模式训练(全 7 相机)带来真实但温和的提升。** 用 k=2 三学生集成作教师、跨全部相机对挖链接(352 条 @ 56%,其中 56 条涉及斜视相机——斜视对齐首次显式训练),从头训 3 个部署学生:全转导 **0.516 → 0.549**(该协议史上最佳;历史起点 0.344 → 累计 +20.5),66.128 查询 +8.1,66.130 的 mAP 0.535→0.585;逐相机不对称仍在。

**论文表述规范 / Reporting rule.** 引用 0.88 必须注明 "leave-out 66.130 协议";全转导协议报 0.549。

---

## 7. 半监督标签效率(遗留底座)/ Label Efficiency (legacy base)

**图 4 / Figure 4:`fig4_label_efficiency_en_v1.png`。** 登记(enroll)部分身份、其余聚类(ViT-S 骨干、2 种子,较早实验):0% → 0.476,10% → 0.559,25% → 0.694,**50% → 0.835**,100% → 0.850。约 50% 登记追平全监督。

> **注意 / Caveat.** 该曲线为旧底座(ViT-S + 簇对比 + 旧读出),与 §2 的 ViT-B 数字不可直接对齐;新无监督底座的 rank-5(0.957)已超过旧半监督 50% 登记的 rank-5(0.932)。相机就绪版应在新底座上以 3–5 种子重跑。

---

## 8. 总体结论 / Overall Conclusions

1. **检索协议本身是第一大杠杆**(骨干读出+白化+排名融合+TTA):零重训 +21.5 分。
2. **跨相机关联质量是无监督的真瓶颈**;攻克它的钥匙是**逐级容噪**:代理级损失 → 成对(非传递)高精度链接 → 固定标签蒸馏 → 种子集成。四次失败与两次成功共同划出这条边界。
3. **单种子方差(±0.11)是无监督评测的头号陷阱**;固定标签蒸馏将其压到 ±0.03,集成给出可复现头条。
4. **蒸馏阶梯一轮收敛**;过训练是自训练的默认失败模式,早停必要。
5. **协议稳健性必须验证**:0.883 仅在基准协议成立;全转导 0.549(史上最佳);有监督的均匀强势证明剩余差距可归因于方法。

---

## 9. 诚实性声明 / Honesty Notes

1. 所有设计决策在基准协议上调优(k=2 之选经 3 种子交叉验证且有先验机理,但元过拟合风险存在);
2. n=163,±1 查询 ≈ 0.6 分;
3. 部署代价:头条配置需 5–10 个模型各嵌入一次(可蒸馏为单模型,未做);
4. oracle 数字(如剔除坏种子的 0.804)仅作上界参考,**不可宣称**;
5. 半监督曲线为遗留底座(§7)。

---

## 10. 图表清单(英文版)/ Figure List (English-only)

| 图 | 文件 | 内容 |
|---|---|---|
| Fig. 1 | `figures/fig1_journey_en_v1.png` | 无监督之旅七级台阶(rank-1 0.503→0.883,全监督虚线)|
| Fig. 2 | `figures/fig2_ensemble_curve_en_v1.png` | 种子集成规模曲线(全子集,均值±std)|
| Fig. 3 | `figures/fig3_percamera_en_v1.png` | 逐相机协议扫描(留出/部署/有监督三系列)|
| Fig. 4 | `figures/fig4_label_efficiency_en_v1.png` | 半监督标签效率(遗留 ViT-S,已标注)|
| Fig. 5 | `figures/fig5_methods_en_v1.png` | 12 方法族得失(发散条形)|
| Fig. 6 | `figures/fig6_pipeline_en_v3.png` | **方法总览 / method overview**:输入结构 → CAP 自训练 → 集成教师+蒸馏学生 → 推理栈,四段式流水线,关键数字标注 |
| Fig. 7 | `figures/fig7_mechanisms_en_v3.png` | **三个核心机制示意 / mechanism demos**:(a) CAP 代理隔离错误合并;(b) 成对链接 vs 传递合并;(c) 稀疏重叠拓扑作为免费监督 |

交互式双语结果面板(非论文用):https://claude.ai/code/artifact/5e6f220d-fa70-4236-b3b0-e889a8b3f1ab

---

## 术语表 / Glossary of Abbreviations

| 缩写 | 全称(英)| 全称(中)|
|---|---|---|
| Re-ID | Re-Identification | 重识别 |
| SSL | Self-Supervised Learning | 自监督学习 |
| DINOv2 | self-DIstillation with NO labels, v2 | 无标签自蒸馏(第 2 版)|
| ViT / ViT-S / ViT-B | Vision Transformer / Small / Base | 视觉 Transformer / 小 / 基础 |
| CLS token | Classification token | 分类标记 |
| CAP | Camera-Aware Proxies | 相机感知代理 |
| O2CAP | Offline-Online associated Camera-Aware Proxies | 离线-在线关联相机感知代理 |
| GeM | Generalized Mean (pooling) | 广义均值池化 |
| BNNeck | Batch Normalization Neck | 批归一化颈 |
| CC | Camera-wise Centering | 逐相机中心化 |
| PCA / PCAW | Principal Component Analysis / PCA Whitening | 主成分分析 / 主成分白化 |
| RR | Re-Ranking | 重排序 |
| CA-Jaccard | Camera-Aware Jaccard (distance) | 相机感知 Jaccard 距离 |
| RRF | Reciprocal Rank Fusion | 倒数排名融合 |
| ST | Spatio-Temporal | 时空 |
| TTA | Test-Time Augmentation | 测试期增强 |
| OT | Optimal Transport | 最优传输 |
| kNN / mutual kNN | (mutual) k-Nearest Neighbours | (互)k 最近邻 |
| DBSCAN | Density-Based Spatial Clustering of Applications with Noise | 基于密度的噪声应用空间聚类 |
| IICS | Intra-Inter Camera Similarity | 相机内-相机间相似度 |
| CMC | Cumulative Matching Characteristics | 累积匹配特征曲线 |
| mAP | mean Average Precision | 平均精度均值 |
| Rank-1 / Rank-5 | Rank-1 / Rank-5 accuracy | 首位命中率 / 前五命中率 |
| AQE | Alpha Query Expansion | α 查询扩展 |
| DBA | DataBase-side Augmentation | 数据库端增强 |
| AMP | Automatic Mixed Precision | 自动混合精度 |
| tracklet | — | 轨迹片段 |
| query / gallery | — | 查询集 / 图库集 |
| cannot-link | — | 不可连接(约束)|
| holdout / deployment mode | — | 留出模式 / 部署模式(是否将评测相机纳入无标签训练)|
| oracle | — | 先知上界(借助真值做选择的参考值,不可宣称)|

---

## 相关文件 / Related Files

- **制胜方法 / winning method:** `vitb_unsup_cap.py`(CAP),`eval_cap_ensemble.py` / `cap_ens_curve.py`(集成与曲线),`make_distill_labels.py` / `distill_diag.py`(教师标签诊断),`vitb_unsup_distill.py`(学生),`fuse_student.py` / `students_only2.py`(融合)
- **协议验证 / validation:** `validate_protocols.py`,`validate_supervised.py`,`vitb_unsup_deploy.py` + `validate_deploy.py`
- **检索期杠杆 / inference levers:** `new_levers*.py`,`vitb_tta.py`,`cowreid/st_inference.py`
- **负面结果 / negatives:** `vitb_unsup_gembn.py`,`vitb_unsup_boot.py`,`vitb_unsup_strict.py` + `strict_mine_diag.py`,`part_match.py`,`consensus_ens.py`
- **图表 / figures:** `make_figures_en.py`(结果图,英文学术版),`make_methods_figs_en.py`(方法总览与机制示意图),`make_figures.py`(双语版)
- **结果存档 / archives:** `artifacts2/*.json`;嵌入缓存 `_vitb_*_emb_*.npz`;checkpoint `_vitb_cap_s{0..4}_ckpt.pt`,`_vitb_dst_s{5..9}_ckpt.pt`,`_vitb_dep_s{10..12}_ckpt.pt`
