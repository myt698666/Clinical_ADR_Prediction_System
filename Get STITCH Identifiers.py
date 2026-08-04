import pandas as pd
import os
import pubchempy as pcp
import time
from tqdm import tqdm

# --- 原作者的文件路径和读取配置 ---
file1 = "All_filtered_chemicals_file.tsv"
file2 = "Drug InChi Keys/SSRIs Inhibitors Inchi.csv"
output_folder = "STITCH_Identifiers"
output_file = os.path.join(output_folder, "matched_chemicals.csv")

os.makedirs(output_folder, exist_ok=True)

df1 = pd.read_csv(file1, sep="\t")
df2 = pd.read_csv(file2)
# ----------------------------------

print("正在启动高级药物名称匹配引擎 (PubChem)...")

# ================= 魔法第一步：提取并去重 =================
# 把两个表里的药物名字提取出来，去掉空值，合并在一起，然后去重
unique_names_1 = df1['name'].dropna().unique()
unique_names_2 = df2['Drug'].dropna().unique()

# set() 会自动把重复的名字剔除
all_unique_names = list(set(unique_names_1) | set(unique_names_2))
print(f"太棒了！去重后，真正需要联网查询的独立药物名称只有: {len(all_unique_names)} 个！")

# ================= 魔法第二步：建立备忘录（字典） =================
cid_dictionary = {}

def get_standard_cid(drug_name):
    try:
        clean_name = str(drug_name).lower().strip()
        results = pcp.get_compounds(clean_name, 'name')
        if results:
            return results[0].cid
        time.sleep(0.2)
    except Exception:
        pass
    return None

print("开始查字典，请稍候...")
# 只对去重后的少量名单进行查询！
for name in tqdm(all_unique_names):
    cid = get_standard_cid(name)
    if cid is not None:
        # 查到了就记录在备忘录里： 名字 -> ID
        cid_dictionary[name] = cid

# ================= 魔法第三步：瞬间映射 =================
print("查询完毕！正在将结果瞬间映射回原本的三万多行数据中...")
test_df1 = df1.copy()
test_df2 = df2.copy()

# .map() 函数会拿着 3 万多行名字去查我们的备忘录，不到 1 秒钟就填好了！
test_df1['CID'] = test_df1['name'].map(cid_dictionary)
test_df2['CID'] = test_df2['Drug'].map(cid_dictionary)

# 剔除查不到的，然后合并
test_df1_valid = test_df1.dropna(subset=['CID'])
test_df2_valid = test_df2.dropna(subset=['CID'])

merged_df = pd.merge(test_df1_valid, test_df2_valid, on='CID', how='inner')
result_df = merged_df[['chemical', 'Drug']]

result_df.to_csv(output_file, index=False)

print(f"\n--- 魔法优化版全量匹配完毕 ---")
print(f"我们成功挽救并匹配出的数据量: {len(result_df)} 个！")
print(f"结果已安全保存至: {output_file}")