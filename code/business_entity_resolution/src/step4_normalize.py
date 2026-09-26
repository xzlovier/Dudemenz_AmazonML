import re
import unicodedata
import pandas as pd
import os
import gc

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_name(name):
    name = normalize_text(name)
    
    # Legal suffix and abbreviation dictionary
    replacements = {
        r'\bcorp\b': 'corporation',
        r'\bpvt\b': 'private',
        r'\bltd\b': 'limited',
        r'\binc\b': 'incorporated',
        r'\bco\b': 'company',
        r'\bllc\b': 'limited liability company',
        r'\band\b': 'and'  # & was removed by punctuation strip, so we might need to handle & earlier if we wanted it
    }
    
    for pat, repl in replacements.items():
        name = re.sub(pat, repl, name)
        
    return re.sub(r'\s+', ' ', name).strip()

def normalize_address(address, country):
    addr = normalize_text(address)
    
    # Abbreviation dictionary
    replacements = {
        r'\brd\b': 'road',
        r'\bst\b': 'street',
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bdr\b': 'drive',
        r'\bln\b': 'lane',
        r'\bct\b': 'court',
        r'\bpl\b': 'place',
        r'\bsq\b': 'square',
        r'\bste\b': 'suite',
        r'\bapt\b': 'apartment'
    }
    
    for pat, repl in replacements.items():
        addr = re.sub(pat, repl, addr)
        
    addr = re.sub(r'\s+', ' ', addr).strip()
    
    # Country-aware branching for PIN/ZIP
    pin = ""
    if country == 'US':
        # US Zip code: 5 digits, optionally followed by 4 digits
        match = re.search(r'\b(\d{5}(?:-\d{4})?)\b', address)
        if match:
            pin = match.group(1)
    elif country == 'India':
        # India PIN code: 6 digits
        match = re.search(r'\b(\d{6})\b', address)
        if match:
            pin = match.group(1)
    else: # France or others
        # France postal code: 5 digits
        match = re.search(r'\b(\d{5})\b', address)
        if match:
            pin = match.group(1)
            
    return addr, pin

def process_file(in_path, out_path):
    print(f"Processing {in_path}...")
    # Use chunking to save memory
    chunk_size = 500000
    first_chunk = True
    
    for chunk in pd.read_csv(in_path, sep="\t", dtype=str, chunksize=chunk_size):
        chunk['business_name'] = chunk['business_name'].fillna('')
        chunk['business_address'] = chunk['business_address'].fillna('')
        chunk['country'] = chunk['country'].fillna('')
        
        chunk['norm_name'] = chunk['business_name'].apply(normalize_name)
        
        # apply address normalization
        addr_pin = chunk.apply(lambda row: normalize_address(row['business_address'], row['country']), axis=1)
        chunk['norm_address'] = addr_pin.apply(lambda x: x[0])
        chunk['extracted_pin'] = addr_pin.apply(lambda x: x[1])
        
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        chunk.to_csv(out_path, sep="\t", index=False, mode=mode, header=header)
        first_chunk = False
        gc.collect()
        
    print(f"Finished {out_path}")

if __name__ == "__main__":
    # Point this to where Kaggle saved your uploaded dataset
    data_dir = "/kaggle/input/datasets/sameerbaranwal/amazon-ml-hackathon-dataset" 
    
    # Kaggle's writable output directory
    out_dir = "/kaggle/working/normalized_dataset"
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)
    
    # Process the TEST files (since you want the final submission)
    process_file(os.path.join(data_dir, "test_source1.tsv"), 
                 os.path.join(out_dir, "test_source1.tsv"))
    process_file(os.path.join(data_dir, "test_source2.tsv"), 
                 os.path.join(out_dir, "test_source2.tsv"))
    process_file(os.path.join(data_dir, "test_source3.tsv"), 
                 os.path.join(out_dir, "test_source3.tsv"))
    
    print("Normalizers loaded and test normalization completed successfully.")
