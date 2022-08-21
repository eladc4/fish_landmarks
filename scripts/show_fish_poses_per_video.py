import os
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.dataset_2d import Dataset2D
from datasets.dataset_2d import show_image_with_keypoints

# print image per video ##########################################################
valid_dataset = Dataset2D(reutrn_target_as_heatmaps=False,
                          transform=transforms.Compose([transforms.ToTensor()]),
                          crop_landmarks=True, train=False, val_split=None,
                          crop_ratio=2.0)

data_loader = DataLoader(
    valid_dataset,
    batch_size=20,
    shuffle=False,
    num_workers=8,
)

results_folder = os.path.join(os.environ.get('SWAT'), 'results', os.environ.get('USER'),
                                  'fish_landmarks', 'fish_poses')
print('results folder:', results_folder)
os.makedirs(results_folder, exist_ok=True)

vid_path = ''
for i, (data, target, target_weight, pp) in enumerate(data_loader):
    for j in range(data.size(0)):
        current_vid_path = pp[0][j].split('/')[-2]
        if current_vid_path != vid_path:
            vid_path = current_vid_path
            _label = target.cpu().detach().numpy()[j, ...]
            img = show_image_with_keypoints(np.uint8(data.cpu().detach().numpy()[j, 0, ...] * 255), _label)
            img.save(os.path.join(results_folder, f'{vid_path}.jpg'))
            print(f'written image for {vid_path}')

