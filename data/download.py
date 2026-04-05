import os
from huggingface_hub import hf_hub_download, list_repo_files

def setup_dataset():
    repo_id = "Mars-Dingdang/DL-AdaptiveOptics"
    base_data_dir = "../data"
    lmdb_dir = os.path.join(base_data_dir, "turbulence_seq_nwpu_mild50_lmdb")
    os.makedirs(base_data_dir, exist_ok=True)
    os.makedirs(lmdb_dir, exist_ok=True)
    print(f"Downloading dataset from Hugging Face Hub repository '{repo_id}' to '{lmdb_dir}'...")
    
    try:
        all_files = list_repo_files(repo_id)
        for file_path in all_files:
            if file_path.endswith(".zip"):
                target_dir = base_data_dir
                print(f"Downloading '{file_path}' to '{target_dir}'...")
            elif file_path.endswith(".mdb"):
                target_dir = lmdb_dir
                print(f"Downloading '{file_path}' to '{target_dir}'...")
            else: 
                continue
            
            downloaded_path = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset", local_dir=target_dir, local_dir_use_symlinks=False)
            print(f"Downloaded '{file_path}' to '{downloaded_path}'")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        
if __name__ == "__main__":
    setup_dataset()