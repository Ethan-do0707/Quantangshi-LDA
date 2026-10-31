# Period Metadata Representativeness Audit

本目录审计 author-period KNOWN/UNKNOWN 子样本的覆盖率、作者产量结构、20维 Topic probability、dominant topic、topic type mass、entropy、CBDB identity resolution 与 STRICT/CORE/EXPANDED 样本代表性。

权威诗级输入：`D:/中大中文系工作相关/論文發表與投稿/全唐詩LDA/historical_metadata/master/C_K20_literary_history_master_with_author_history.csv`  
SHA-256：`93fa45a5d848d0708fd1f4fb44289c26d8e98b7eec5b56bc440dc0de95de801f`

诗级输入缺少 `author_status` 与 `cbdb_match_status`，故只按唯一 `author_id_internal` 从 `D:/中大中文系工作相关/論文發表與投稿/全唐詩LDA/historical_metadata/master/QTS_author_historical_metadata.csv` 确定性连接这两列；不修改权威 CSV。

本任务是 missingness/representativeness audit，不比较初盛中晚唐，不重新训练 LDA，不修改 CBDB 或 period，不进行重加权或时期填补。
