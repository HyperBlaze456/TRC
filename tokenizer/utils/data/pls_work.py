import os
import datasets
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login


# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Set token globally for HuggingFace
if hf_token:
    print(f"Setting HF token (first 8 chars): {hf_token[:8]}...")
    try:
        login(token=hf_token)
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("HF token set successfully")
    except Exception as e:
        print(f"Warning: Failed to login with HF token: {e}")
else:
    print("Warning: No HF_TOKEN found in environment")

ds = load_dataset(
    "amphion/Emilia-Dataset",
    "default",
    split="train",
    streaming=True
)


class BatchedDatasetWrapper:
    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size
    
    def __iter__(self):
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        # Yield remaining samples if any
        if batch:
            yield batch


# Example usage
batched_ds = BatchedDatasetWrapper(ds, batch_size=16)

# Test iteration
for i, batch in enumerate(batched_ds):
    print(f"Batch {i}: {len(batch)} samples")
    # Print first sample keys from the batch
    if i == 0:
        print(f"Sample keys: {list(batch[0].keys())}")
    
    # Stop after a few batches for testing
    if i >= 2:
        break



# Test native HuggingFace batching
print("\n--- Testing native HuggingFace .batch() method ---")
native_batched = ds.batch(batch_size=16)

for i, batch in enumerate(native_batched):
    print(f"Native Batch {i}:")
    print(f"  Type: {type(batch)}")
    print(f"  Keys: {list(batch.keys())}")
    # Check the structure - batch should be a dict with lists as values
    for key in list(batch.keys())[:1]:  # Just check first key
        print(f"  {key} shape: {type(batch[key])}, length: {len(batch[key]) if isinstance(batch[key], list) else 'N/A'}")
    
    if i >= 2:
        break

