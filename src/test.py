import torch

# 👉 PUT YOUR ACTUAL PATH HERE
pt_path = r"C:\Users\HP\Desktop\MIL\features\TCGA-A6-2676-11A-01-TS1.a2b894be-974e-4031-9b67-9bbe98e37fe4.pt"

# Load file
data = torch.load(pt_path, map_location="cpu")

print("\n===== BASIC INFO =====")
print("TYPE:", type(data))

# =========================
# CASE 1: DICTIONARY
# =========================
if isinstance(data, dict):
    print("\nKEYS:", data.keys())

    for key in data.keys():
        value = data[key]
        print(f"\n--- {key} ---")
        print("Type:", type(value))
        
        # If tensor, print shape
        if isinstance(value, torch.Tensor):
            print("Shape:", value.shape)
            print("Dtype:", value.dtype)
            print("Sample values:", value[:5])  # first few entries
        
        else:
            print("Value:", value)

# =========================
# CASE 2: TENSOR
# =========================
elif isinstance(data, torch.Tensor):
    print("\nTensor Shape:", data.shape)
    print("Dtype:", data.dtype)
    print("Sample values:", data[:5])

# =========================
# CASE 3: LIST / TUPLE
# =========================
elif isinstance(data, (list, tuple)):
    print("\nLength:", len(data))
    for i, item in enumerate(data):
        print(f"\n--- Item {i} ---")
        print("Type:", type(item))
        try:
            print("Shape:", item.shape)
        except:
            print("Value:", item)

# =========================
# UNKNOWN TYPE
# =========================
else:
    print("\nUnknown structure")
    print(data)