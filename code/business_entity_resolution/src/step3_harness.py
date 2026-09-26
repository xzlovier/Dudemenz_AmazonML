import pandas as pd
import numpy as np

def compute_metrics(ground_truth_df, candidates_df, matches_df):
    """
    ground_truth_df: DataFrame with ['source1_entity_id', 'matched_entity_ids', 'country', 'match_count']
    candidates_df: DataFrame with ['source1_entity_id', 'candidate_entity_ids']
    matches_df: DataFrame with ['source1_entity_id', 'matched_entity_ids']
    """
    # 1. Blocking recall and candidate set size
    # We evaluate this over all S1 entities that HAVE a match in ground truth
    merged_cand = ground_truth_df.merge(candidates_df, on='source1_entity_id', how='left')
    merged_cand['candidate_entity_ids'] = merged_cand['candidate_entity_ids'].fillna("")
    
    total_true_matches = 0
    total_found_candidates = 0
    total_candidates_generated = 0
    
    for _, row in merged_cand.iterrows():
        if not row['matched_entity_ids']:
            true_set = set()
        else:
            true_set = set(row['matched_entity_ids'].split(','))
            
        if not row['candidate_entity_ids']:
            cand_set = set()
        else:
            cand_set = set(row['candidate_entity_ids'].split(','))
            
        total_true_matches += len(true_set)
        total_found_candidates += len(true_set.intersection(cand_set))
        total_candidates_generated += len(cand_set)
        
    blocking_recall = total_found_candidates / total_true_matches if total_true_matches > 0 else 1.0
    avg_candidates = total_candidates_generated / len(ground_truth_df)
    
    print(f"Blocking Recall: {blocking_recall:.4f}")
    print(f"Avg Candidates per S1: {avg_candidates:.2f}")

    # 2. F0.5 Macro Average
    merged_match = ground_truth_df.merge(matches_df, on='source1_entity_id', how='left', suffixes=('_true', '_pred'))
    merged_match['matched_entity_ids_pred'] = merged_match['matched_entity_ids_pred'].fillna("")
    
    f05_scores = []
    
    for _, row in merged_match.iterrows():
        true_str = row['matched_entity_ids_true']
        pred_str = row['matched_entity_ids_pred']
        
        true_set = set(true_str.split(',')) if true_str else set()
        pred_set = set(pred_str.split(',')) if pred_str else set()
        
        if len(true_set) == 0:
            if len(pred_set) == 0:
                f05_scores.append(1.0)
            else:
                f05_scores.append(0.0)
        else:
            tp = len(true_set.intersection(pred_set))
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            
            if tp == 0:
                f05_scores.append(0.0)
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                f05_scores.append(f05)
                
    macro_f05 = np.mean(f05_scores)
    print(f"Macro F0.5 Score: {macro_f05:.4f}")
    
    return {
        "blocking_recall": blocking_recall,
        "avg_candidates": avg_candidates,
        "macro_f05": macro_f05
    }

if __name__ == "__main__":
    print("Testing the harness with a trivial baseline (predict everything empty)...")
    gt = pd.DataFrame({
        'source1_entity_id': ['S1-1', 'S1-2', 'S1-3'],
        'matched_entity_ids': ['S2-1,S3-1', '', 'S2-2']
    })
    
    cands = pd.DataFrame({
        'source1_entity_id': ['S1-1', 'S1-2', 'S1-3'],
        'candidate_entity_ids': ['S2-1,S3-1,S2-3', '', 'S2-2']
    })
    
    matches = pd.DataFrame({
        'source1_entity_id': ['S1-1', 'S1-2', 'S1-3'],
        'matched_entity_ids': ['', '', '']
    })
    
    print("Zero baseline:")
    compute_metrics(gt, cands, matches)
    
    print("\nPerfect baseline:")
    compute_metrics(gt, cands, gt)

