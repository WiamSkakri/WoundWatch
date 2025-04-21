from dataset import WoundSegmentationDataset
from torchvision import transforms
import matplotlib.pyplot as plt

train_dataset = WoundSegmentationDataset(
    image_dir="../data/train/Images",
    mask_dir="../data/train/Masks",
    transform_img=transforms.ToTensor(),
    transform_mask=transforms.ToTensor()
)

image, mask = train_dataset[0]

plt.subplot(1, 2, 1)
plt.imshow(image.permute(1, 2, 0))
plt.title("Image")

plt.subplot(1, 2, 2)
plt.imshow(mask.squeeze(), cmap="gray")
plt.title("Mask")
plt.show()

