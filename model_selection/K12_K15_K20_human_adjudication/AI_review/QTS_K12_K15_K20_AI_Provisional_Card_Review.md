## 1. Objective

对 47 张匿名主题卡进行 AI 暂定语义审读，保持 Researcher_* 字段原样；结果只作为研究者复核材料。

## 2. Card Inventory

HR001–HR047 共 47 张均存在，Top30、Top20 代表诗与作者平衡 Top10 均完整。详见 `AI_review_card_inventory.csv`。

## 3. Blind Review Protocol

按 HR001→HR047 审读并在解盲前冻结；冻结 SHA256：`d9f17b9928e3ac58190673de59e15e2abb99b1ddf10cd57bb679d96db75cb9c9`。本轮未提前读取真实 K、映射或量化指标。受同一对话早期上下文曾暴露少量旧 AI 内容影响，不宣称对旧 AI 的完全独立隔离。

## 4. Review Rubric

八项 1–5 分包括词语凝聚、诗例凝聚、词诗一致、可命名、文学可解释、语义特定性、过宽度与内部混合度；后两项高分代表风险较高。类型限定 CLEAR/BROAD/MIXED/NOISE。

## 5. 47-Topic Review Completion

47/47 主题的 15 个 AI_Provisional_* 字段均已填写；Needs_Splitting=YES 均列出疑似子主题，MIXED/NOISE 均有证据与说明。

## 6. K12 Summary

CLEAR 7，BROAD 2，MIXED 2，NOISE 1；文学可解释性均值 4.5833，过宽度均值 2.3333，Needs_Splitting=YES 3/12。

## 7. K15 Summary

CLEAR 8，BROAD 5，MIXED 1，NOISE 1；文学可解释性均值 4.4，过宽度均值 2.3333，Needs_Splitting=YES 2/15。

## 8. K20 Summary

CLEAR 10，BROAD 4，MIXED 4，NOISE 2；文学可解释性均值 4.2，过宽度均值 2.6，Needs_Splitting=YES 6/20。

## 9. K12 Broad / Noise Challenge

复核对象：HR002/T6 RETAIN_WITH_CAUTION；HR026/T10 RETAIN_WITH_CAUTION；HR035/T1 SPLIT_CANDIDATE；HR037/T4 UNSTABLE_TOPIC；HR045/T7 SPLIT_CANDIDATE。挑战意见另存于解盲主题表。

## 10. K12→K15 Split Review

8 个分裂候选：MEANINGFUL_SPLIT 5，MINOR_REFINEMENT 1，FRAGMENTATION 1，UNCERTAIN 1。逐事件依据见 `AI_split_merge_review.csv`。

## 11. K15→K20 Split Review

8 个分裂候选：MEANINGFUL_SPLIT 4，MINOR_REFINEMENT 2，FRAGMENTATION 2，UNCERTAIN 0。K20 虽增加若干清晰细分，也生成混合残余。

## 12. Fragmentation Review

15 对候选中 TRUE_FRAGMENTATION 0，MEANINGFUL_SEPARATION 14，UNCERTAIN 1。判断基于词与诗的文学功能，不以 cosine/Jaccard 自动裁决。

## 13. AI Provisional Model Comparison

K12 的清晰主题比例较高但包含宽主题与噪声；K15 在清晰度、混合风险和语义分辨率之间较平衡；K20 提供若干有效细分，但 MIXED/NOISE 比例上升。鉴于旧 AI 隔离限制与二阶段残余风险，状态记为 `AI_MORE_REVIEW_REQUIRED`，不作最终 K 决策。

## 14. Items Requiring Researcher Attention

K12：HR035/T1 奇山瀑布、登临寻仙与清赏；HR037/T4 行旅仕宦、寄赠与古迹杂咏；HR002/T6 宫闺春夜、失宠与相思；HR045/T7 生民疾苦、劝世寓言与历史杂咏；HR020/T8 帝王功德、武成礼祀与朝廷恩命；HR026/T10 山水送别、友朋酬寄与归隐之思。

K15：HR038/T1 仙山宫阙、华清游幸与歌舞华饰；HR027/T6 老境诗酒、闲居退隐与知己酬答；HR032/T8 山居隐逸、仙境访道与禅僧往还；HR046/T9 诗名仕宦、送行宴集与礼仪杂咏；HR019/T11 春花美人、赏花惜春与宫闺愁情；HR013/T2 行路困厄、世道不平与生死悲愤；HR017/T4 关塞骑行、军旅送别与归程；HR033/T14 官居清游、山林归思与僧道雅集；HR007/T15 边塞征战、武功与将士。

K20：HR003/T1 宫廷恩幸、仪仗与华贵物象；HR010/T6 衰老病困中的身世自咏与诗友酬答；HR022/T9 年时迁转、出处送归与杂体题咏；HR011/T13 清光洁性与祥瑞自然物象的赋咏；HR028/T14 清寒云景、庭园题咏与宫苑杂咏；HR005/T15 文章声名、仕宦荣典与传世功业；HR044/T18 清夜声景、寺游送行与杂体物事；HR025/T2 乱世征战、忠义报国与历史悲愤；HR012/T4 春归新岁、江南乡思与惜时；HR001/T7 秋夜月下的离愁、相思与思归。

## 15. Limitations

本结果是单一 AI 的暂定审读，不替代研究者裁决；同一对话早期曾出现少量旧 AI 内容，严格独立盲审受到限制。量化指标未用于修改冻结语义分数，候选关系也不被视为真实谱系。
