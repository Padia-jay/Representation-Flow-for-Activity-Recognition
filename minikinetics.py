import torch
import torch.utils.data as data_utl
import numpy as np
import random
import os
import json
import cv2

class MK(data_utl.Dataset):
    def __init__(self, split_file, root, mode='rgb', length=64, random=True, model='2d', size=112):
        with open(split_file, 'r') as f:
            self.data = json.load(f)
        self.vids = [k for k in self.data.keys()]

        if mode == 'flow':
            new_data = {}
            self.vids = ['flow' + v[3:] for v in self.vids]
            for v in self.data.keys():
                new_data['flow' + v[3:]] = self.data[v]
            self.data = new_data
        
        self.split_file = split_file
        self.root = root
        self.mode = mode
        self.model = model
        self.length = length
        self.random = random
        self.size = size

    def load_video_frames(self, video_path):
        frames = []  # Placeholder for loaded frames

        # OpenCV to load video frames
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        return np.array(frames)

    def __getitem__(self, index):
        vid = self.vids[index]
        cls = self.data[vid]
        video_path = os.path.join(self.root, vid)

        if not os.path.exists(video_path):
            # Return zeros if video not found
            if self.mode == 'flow' and self.model == '2d':
                return np.zeros((3, 20, self.size, self.size), dtype=np.float32), 0
            elif self.mode == 'flow' and self.model == '3d':
                return np.zeros((2, self.length, self.size, self.size), dtype=np.float32), 0

        frames = self.load_video_frames(video_path)

        # Process frames as needed - cropping, resizing, normalization, etc.

        # Convert frames to tensor
        frames_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float()

        return frames_tensor, cls

    def __len__(self):
        return len(self.data.keys())

if __name__ == '__main__':
    train = 'E:/representation-flow-cvpr19-master/representation-flow-cvpr19-master/data/kinetics/minikinetics_train.json'
    val = 'representation-flow-cvpr19-master/data/kinetics/minikinetics_val.json'
    root = 'E:/representation-flow-cvpr19-master/representation-flow-cvpr19-master/data/kineticsS'
    dataset_tr = MK(train, root, length=16, model='2d', mode='flow')
    print(dataset_tr[random.randint(0, 1000)])
