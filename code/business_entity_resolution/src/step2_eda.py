import pandas as pd
import os
import random

data_dir = os.path.join("student_resource", "dataset")

print("Loading data for EDA...")
gt = pd.read_csv(os.path.join(data_dir, "train", "train_ground_truth.tsv"), sep="\t", dtype=str)
s1 = pd.read_csv(os.path.join(data_dir, "train", "train_source1.tsv"), sep="\t", dtype=str)

# 1. Cardinality S1 entities
print("\n--- Cardinality of Matches ---")
gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna("")
def count_matches(x):
    if not x: return 0
    return len(x.split(','))
gt['match_count'] = gt['matched_entity_ids'].apply(count_matches)

# categorization
def categorize(c):
    if c == 0: return "Singleton"
    if c == 1: return "1-match"
    return "many-match"
gt['match_category'] = gt['match_count'].apply(categorize)
print(gt['match_category'].value_counts(normalize=True) * 100)

# 2. Country distribution S1
print("\n--- Country Distribution S1 ---")
print(s1['country'].value_counts(normalize=True) * 100)

# France in test set
test_s1 = pd.read_csv(os.path.join(data_dir, "test", "test_source1.tsv"), sep="\t", dtype=str)
print("\n--- Country Distribution Test S1 ---")
print(test_s1['country'].value_counts(normalize=True) * 100)

if "France" in test_s1['country'].values:
    print("\nFrance addresses sample:")
    print(test_s1[test_s1['country'] == 'France'][['business_address']].head(5))

# 3. Sample True Positive Pairs
print("\n--- Generating 50 True Positive Pairs ---")
s2 = pd.read_csv(os.path.join(data_dir, "train", "train_source2.tsv"), sep="\t", dtype=str, usecols=['entity_id', 'business_name', 'business_address'])
s3 = pd.read_csv(os.path.join(data_dir, "train", "train_source3.tsv"), sep="\t", dtype=str, usecols=['entity_id', 'business_name', 'business_address'])

s2_dict = s2.set_index('entity_id').to_dict('index')
s3_dict = s3.set_index('entity_id').to_dict('index')

gt_with_matches = gt[gt['match_count'] > 0]
sample_gt = gt_with_matches.sample(50, random_state=42)

with open("eda_sample_pairs.txt", "w", encoding="utf-8") as f:
    for idx, row in sample_gt.iterrows():
        s1_id = row['source1_entity_id']
        s1_row = s1[s1['entity_id'] == s1_id].iloc[0]
        f.write(f"S1: {s1_id} | {s1_row['business_name']} | {s1_row['business_address']} | {s1_row['country']}\n")
        
        matches = row['matched_entity_ids'].split(',')
        for m in matches:
            if m.startswith('S2-') and m in s2_dict:
                match_data = s2_dict[m]
                f.write(f" -> {m} (S2) | {match_data['business_name']} | {match_data['business_address']}\n")
            elif m.startswith('S3-') and m in s3_dict:
                match_data = s3_dict[m]
                f.write(f" -> {m} (S3) | {match_data['business_name']} | {match_data['business_address']}\n")
            else:
                f.write(f" -> {m} (NOT FOUND)\n")
        f.write("-" * 80 + "\n")
print("Saved 50 sample pairs to eda_sample_pairs.txt")
