import torch
import torch.utils.data as data_utl
import numpy as np
import random
import os
import cv2

#def video_to_tensor(pic):
#    """Convert a ``numpy.ndarray`` to tensor.
#    Converts a numpy.ndarray (T x H x W x C)
#    to a torch.FloatTensor of shape (C x T x H x W)
#    
#    Args:
#         pic (numpy.ndarray): Video to be converted to tensor.
 #   Returns:
#         Tensor: Converted video.
#    """
#    return torch.from_numpy(pic.transpose((3, 0, 1, 2)))

def video_to_tensor(pic):
    # Ensure the input array has the shape (T x H x W x C)
    if pic.shape[0] != 3:  # If the channels are not the first dimension
        pic = np.moveaxis(pic, -1, 0)  # Adjust to (T x H x W x C) format

    return torch.from_numpy(pic)

class HMDB(data_utl.Dataset):
    def __init__(self, split_file, root, mode='rgb', length=16, model='2d', random=True, c2i={}):
        self.class_to_id = c2i
        self.id_to_class = []
        for i in range(len(c2i.keys())):
            for k in c2i.keys():
                if c2i[k] == i:
                    self.id_to_class.append(k)
        cid = 0
        self.data = []
        self.model = model
        self.size = 112

        with open(split_file, 'r') as f:
            for l in f.readlines():
                if len(l) <= 5:
                    continue
                v, c = l.strip().split(' ')
                v = mode + '_' + v.split('.')[0] + '.mp4'
                if c not in self.class_to_id:
                    self.class_to_id[c] = cid
                    self.id_to_class.append(c)
                    cid += 1
                self.data.append([os.path.join(root, v), self.class_to_id[c]])

        self.split_file = split_file
        self.root = root
        self.mode = mode
        self.length = length
        self.random = random

    def load_video_frames(self, video_path):
        # Replace this with your custom function to load video frames
        # Use libraries like OpenCV or imageio to load frames from a video file
        # Example: Use OpenCV to load video frames

        frames = []  # Placeholder for loaded frames
        # Use your mechanism to load video frames from a file (e.g., using OpenCV)
        # Example:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
             ret, frame = cap.read()
             if not ret:
                 break
             frames.append(frame)
        cap.release()

        # Convert the frames list to a numpy array
        frames_np = np.array(frames)
        return frames_np

    def __getitem__(self, index):
        vid, cls = self.data[index]

        frames = self.load_video_frames(vid)  # Load video frames using your custom function

        # Apply your processing on frames as required for cropping, resizing, and normalization
        # ...

        # Process frames (cropping, resizing, normalization) according to your needs
        # Example:
        # frames = some_processing_function(frames)
        
        if not self.random:
            i = int(round((h-self.size)/2.))
            j = int(round((w-self.size)/2.))
            frames = np.reshape(frames, newshape=(self.length*2, h*2, w*2, 3))[::2,::2,::2,:][:, i:-i, j:-j, :]
        else:
            th = self.size
            tw = self.size
            i = random.randint(0, h - th) if h!=th else 0
            j = random.randint(0, w - tw) if w!=tw else 0
            frames = np.reshape(frames, newshape=(self.length*2, h*2, w*2, 3))[::2,::2,::2,:][:, i:i+th, j:j+tw, :]
            
        if self.mode == 'flow':
            #print(frames[:,:,:,1:].mean())
            #exit()
            # only take the 2 channels corresponding to flow (x,y)
            frames = frames[:,:,:,1:]
            if self.model == '2d':
                # this should be redone...
                # stack 10 along channel axis
                frames = np.asarray([frames[:10],frames[2:12],frames[4:14]]) # gives 3x10xHxWx2
                frames = frames.transpose(0,1,4,2,3).reshape(3,20,self.size,self.size).transpose(0,2,3,1)
            
                
        frames = 1-2*(frames.astype(np.float32)/255)

        if self.model == '2d':
            # 2d -> return TxCxHxW
            return frames.transpose([0,3,1,2]), cls
        # 3d -> return CxTxHxW
        return frames.transpose([3,0,1,2]), cls

        # Convert frames to tensor
        frames_tensor = video_to_tensor(frames)

        return frames_tensor, cls

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    # Usage remains the same
    DS = HMDB
    dataseta = DS('data/hmdb/split1_train.txt', '/ssd/hmdb/', model='2d', mode='flow', length=16)
    dataset = DS('data/hmdb/split1_test.txt', '/ssd/hmdb/', model='2d', mode='rgb', length=16, c2i=dataseta.class_to_id)

    for i in range(len(dataseta)):
        print(dataseta[i][0].shape)
