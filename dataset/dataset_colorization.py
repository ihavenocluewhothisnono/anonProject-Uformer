import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms




class ColorizationDataset(Dataset):
  def __init__(self, root_dir, transform=None):
      """
      Args:
          root_dir (string): Path to folder containing RGB images.
          transform (callable, optional): Transform to be applied to both input (grayscale) and target (RGB).
      """
      self.root_dir = root_dir
      self.transform = transform

      #Collects all the images from the file
      self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]


  def __len__(self):
      return len(self.image_files)

  def __getitem__(self, idx):
      img_path = os.path.join(self.root_dir, self.image_files[idx])
      rgb_img = Image.open(img_path).convert('RGB') 
    cana #converts the RGB image to grayscael as an input
      gray_img = rgb_img.convert('L')

      if self.transform:
          rgb_img = self.transform(rgb_img)
          gray_img = self.transform(gray_img)

      return gray_img, rgb_img, self.image_files[idx], self.image_files[idx]
