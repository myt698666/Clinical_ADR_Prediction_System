import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import os

print("⏳ [1/4] 正在加载基础地基数据 (500维蛋白质特征)...")
# 读取昨晚跑出来的终极矩阵
matrix_file = "STITCH_Identifiers/top_500_protein_drug_interaction_matrix.csv"
protein_matrix = pd.read_csv(matrix_file)

print("🧬 [2/4] 正在读取原始 SMILES 化学结构式...")
# 读取原作者基础表里的化学结构（我们在之前的截图中看到过这个文件）
smiles_file = "Drug InChi Keys/All_drug_Inchi_and_smiles.csv"
smiles_df = pd.read_csv(smiles_file)

# 提取我们那 1100 多个精英药物的 SMILES 结构
elite_drugs_smiles = pd.merge(protein_matrix[['Matched Drug']], smiles_df[['Drug', 'SMILES']],
                              left_on='Matched Drug', right_on='Drug', how='inner')

# 定义降维打击函数：将 SMILES 转化为 2048维的 Morgan 指纹 (ECFP4)
def get_morgan_fingerprint(smiles_string):
    try:
        # 将文本转换为 RDKit 的分子对象
        mol = Chem.MolFromSmiles(str(smiles_string))
        if mol:
            # 提取半径为 2，位数为 2048 的指纹特征
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            return list(fp)
        else:
            return [0] * 2048 # 如果结构式非法，填充 0
    except:
        return [0] * 2048

print("🔬 [3/4] 启动 RDKit 引擎：计算 2048 维分子指纹 (这可能需要十几秒)...")
# 应用函数，生成指纹
fingerprints = elite_drugs_smiles['SMILES'].apply(get_morgan_fingerprint)

# 将提取出的一大坨数据，转化为整齐的 2048 列的表格
fp_df = pd.DataFrame(fingerprints.tolist(), columns=[f'ChemFp_{i}' for i in range(2048)])
fp_df.insert(0, 'Matched Drug', elite_drugs_smiles['Matched Drug'])

print("🔗 [4/4] 正在执行多模态特征融合 (Multi-modal Feature Fusion)...")
# 将【500维的蛋白质靶点】与【2048维的化学指纹】强强联手！
final_fusion_matrix = pd.merge(protein_matrix, fp_df, on='Matched Drug', how='inner')

# 保存这套终极无敌矩阵！
output_path = "STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv"
final_fusion_matrix.to_csv(output_path, index=False)

print("\n🎉 升华大功告成！")
print(f"✅ 融合矩阵已保存至: {output_path}")
print(f"📊 最终你将喂给 AI 的特征维度高达: {final_fusion_matrix.shape[1] - 1} 维！(500个靶点 + 2048个化学特征)")