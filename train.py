from utils.data_loader import XRayDataManager
from model import mamba_former
data_manager = XRayDataManager()
train_loader, val_loader, test_loader = data_manager.create_dataloaders()


# print("\nTesting train loader:")
# for batch_idx, (images, labels, image_names) in enumerate(train_loader):
#     print(f"Batch {batch_idx}: {images.shape}, Labels: {labels[:1]}")
#     if batch_idx == 0:
#         break

# print("\nTesting val loader:")
# for batch_idx, (images, labels, image_names) in enumerate(val_loader):
#     print(f"Batch {batch_idx}: {images.shape}, Labels: {labels[:1]}")
#     if batch_idx == 0:
#         break






