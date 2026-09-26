import pandas as pd
import os

print("Generating dummy candidate_pairs.tsv and matching_results.tsv for validation...")

test_s1_path = os.path.join("dataset", "test", "test_source1.tsv")
s1_ids = pd.read_csv(test_s1_path, sep="\t", usecols=["entity_id"])['entity_id']

# Create empty candidate_pairs.tsv and matching_results.tsv
out_cands = []
out_matches = []

for s1 in s1_ids:
    out_cands.append(f"{s1}\t\n")
    out_matches.append(f"{s1}\t\n")

with open(os.path.join("output", "candidate_pairs.tsv"), "w") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    f.writelines(out_cands)

with open(os.path.join("output", "matching_results.tsv"), "w") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    f.writelines(out_matches)

print("Generated files in output/ folder.")
