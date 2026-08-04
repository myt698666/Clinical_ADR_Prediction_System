import pandas as pd
import os
from fuzzywuzzy import process
from tqdm import tqdm

# 激活 pandas 的进度条功能
tqdm.pandas()

# --- 1. 读取原始数据 ---
file1 = "All_filtered_chemicals_file.tsv"

# 【终极修改】：正式换成我们刚刚挖出来的全量药物大名单！
file2 = "extracted_links_and_names.csv"

output_folder = "STITCH_Identifiers"
output_file = os.path.join(output_folder, "matched_chemicals_hybrid.csv")
os.makedirs(output_folder, exist_ok=True)

df1 = pd.read_csv(file1, sep="\t")
df2 = pd.read_csv(file2)

# 【核心适配】：把 'Drug Name' 自动重命名为 'Drug'，完美兼容后续所有代码
if 'Drug Name' in df2.columns:
    df2 = df2.rename(columns={'Drug Name': 'Drug'})

# 清理列名：去空格、转小写
df1['name_clean'] = df1['name'].astype(str).str.lower().str.strip()
df2['drug_clean'] = df2['Drug'].astype(str).str.lower().str.strip()

print(f"输入成功！当前待匹配的黄卡药物总数: {len(df2)} 个")
print("第一阶段：执行精确匹配，稳拿保底数据...")

# --- 2. 完美精确匹配 ---
exact_match_df = pd.merge(df1, df2, left_on='name_clean', right_on='drug_clean', how='inner')
print(f"✅ 第一阶段精确匹配成功: {len(exact_match_df)} 个！")

# --- 3. 智能打捞落网之鱼 ---
matched_ycs_drugs = exact_match_df['drug_clean'].unique()
matched_stitch_drugs = exact_match_df['name_clean'].unique()

unmatched_df2 = df2[~df2['drug_clean'].isin(matched_ycs_drugs)].copy()
unmatched_df1 = df1[~df1['name_clean'].isin(matched_stitch_drugs)].copy()
stitch_names_pool = unmatched_df1['name_clean'].dropna().unique().tolist()

print(f"🔍 发现还有 {len(unmatched_df2)} 个药物由于名字微小差异未能匹配。")
print("第二阶段：启动本地 AI 模糊匹配引擎 (FuzzyWuzzy)...")

def fuzzy_match_drug(target_name):
    # 设定相似度阈值为 88 分，允许微小的拼写、单复数或空格差异
    match_result = process.extractOne(target_name, stitch_names_pool)
    if match_result and match_result[1] >= 88:
        return match_result[0]
    return None

print("⏳ 正在进行高强度智能比对（完全在本地运行，不消耗网络，不惧封锁）...")
unmatched_df2['fuzzy_match_name'] = unmatched_df2['drug_clean'].progress_apply(fuzzy_match_drug)

# 捞回成功的数据
fuzzy_success_df2 = unmatched_df2.dropna(subset=['fuzzy_match_name'])
fuzzy_match_df = pd.merge(unmatched_df1, fuzzy_success_df2, left_on='name_clean', right_on='fuzzy_match_name', how='inner')

print(f"✅ 第二阶段模糊匹配成功挽救: {len(fuzzy_match_df)} 个！")

# --- 4. 终极大会师 ---
final_merged_df = pd.concat([exact_match_df, fuzzy_match_df], ignore_index=True)
result_df = final_merged_df[['chemical', 'Drug']].drop_duplicates()
result_df.to_csv(output_file, index=False)

print(f"\n🎉 --- 全量匹配圆满结束 ---")
print(f"原作者旧逻辑剩下: 1248 个左右")
print(f"我们的新算法最终抢救并获取的总数据量: {len(result_df)} 个！")
print(f"结果已安全保存至: {output_file}")