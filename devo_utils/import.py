import h5py

def print_attrs(name, obj):
    print(f"{name}:")
    for key, val in obj.attrs.items():
        print(f"    {key}: {val}")

def print_dataset(name, dataset):
    print(f"Dataset {name}:")
    data = dataset[()]
    print(data)

def print_group(name, group):
    print(f"Group {name}:")
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            print_dataset(key, item)
        elif isinstance(item, h5py.Group):
            print_group(key, item)

filename = "/home/ebru/DEVO/circles_darker_v_0.10_0/bag_converted.h5"

with h5py.File(filename, "r") as f:
    print("Root keys:", list(f.keys()))
    f.visititems(print_attrs)
    for key in f.keys():
        item = f[key]
        if isinstance(item, h5py.Dataset):
            print_dataset(key, item)
        elif isinstance(item, h5py.Group):
            print_group(key, item)

