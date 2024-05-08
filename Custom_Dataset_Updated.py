def normalize_image(image):
    """
    Normalize image to the range [0, 1].
    """
    image = image - np.min(image)
    return image / np.max(image)

def create_3D_slice(patient_dir):
  patient_slice = []

  index_list = random.sample(range(len(os.listdir(patient_dir))),28)

  for index in index_list:
    dcm_file = os.listdir(patient_dir)[index]
    dcm_path = os.path.join(patient_dir, dcm_file)
    dcm = pydicom.read_file(dcm_path, force= True)
    patient_slice.append(dcm)

  patient_slice = sorted(patient_slice, key=lambda s: s.SliceLocation)  # sort slices by location
  slice_3d = []
  for i in range(len(patient_slice)):       ## Sample slices with step=3
      gray_image = Image.fromarray(patient_slice[i].pixel_array)
      if gray_image.mode == 'I;16':
        gray_image = gray_image.convert('I')
      resize_image = transforms.Resize((224,224))(gray_image)
      resize_image_array = np.array(resize_image)
      img_tensor = torch.from_numpy(normalize_image(resize_image_array)).float()
      img_tensor = img_tensor.unsqueeze(0)
      slice_3d.append(img_tensor)  ## Resize to (224,224) first
  slice_3d = torch.stack(slice_3d, dim=-1)
  age = list(patient_slice[0].PatientAge)
  age = np.array([int(''.join(age[:-1]))])
  age = torch.from_numpy(age).float()

  return slice_3d, age

class PatientDataset(Dataset):
  def __init__(self, based_dir, transform=None):
    self.based_dir = based_dir
    self.transform = transform
    #self.ageMark = create_age_mark(based_dir)

  def __len__(self):
    return len(os.listdir(self.based_dir))

  def __getitem__(self, index):
    dir_list = os.listdir(self.based_dir)
    patient_dir = os.path.join(self.based_dir, dir_list[index])
    sample, agemark = create_3D_slice(patient_dir)

    if self.transform:
      sample = self.transform(sample)
      agemark = self.transform(agemark)
    return sample, agemark

# build dataset
patient_dataset = PatientDataset(based_dir="/content/drive/MyDrive/Experiment_Dataset/")
train_set, valid_set = random_split(patient_dataset, [0.8, 0.2])
train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
for sample, age in train_dataloader:
  print(sample.size())
  print(age.size())
  break
