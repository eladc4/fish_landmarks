
import cv2
import numpy as np
from PIL import Image

from scripts.gen_2d_dataset import show_image_with_keypoints


def get_min_max_landmarks(_landmarks):
    nonan_landmarks = _landmarks[~np.isnan(_landmarks[:, 0]), :]
    _min_x, _max_x = nonan_landmarks[:, 0].min(), nonan_landmarks[:, 0].max()
    _min_y, _max_y = nonan_landmarks[:, 1].min(), nonan_landmarks[:, 1].max()
    return _min_x, _max_x, _min_y, _max_y


class FlipLeftRight:
    def __init__(self, flip_prob=0.5, landmark_flip=(0, 1, 2, 3, 4, 5, 7, 6, 9, 8), num_input_images=1):
        self.flip_prob = flip_prob
        if num_input_images > 1:
            landmark_flip = tuple((np.array(landmark_flip).reshape((1, -1)) +
                                   (np.arange(num_input_images) * len(landmark_flip)).reshape((-1, 1))).flatten())
        self.landmark_flip = landmark_flip

    def __call__(self, np_image, landmarks):
        if np.random.random() < self.flip_prob:
            np_image = np.fliplr(np_image)
            landmarks = landmarks[self.landmark_flip, :]
            landmarks[:, 0] = np_image.shape[1] - landmarks[:, 0]
        return np_image, landmarks


class RandomRotateScale:
    def __init__(self, max_angle=None, scale_limits=None):
        self.max_angle = max_angle
        self.scale_limits = scale_limits

    def __call__(self, np_image, landmarks):
        if self.max_angle or self.scale_limits:
            min_x, max_x, min_y, max_y = get_min_max_landmarks(landmarks)
            scale = np.random.uniform(low=self.scale_limits[0], high=self.scale_limits[1]) if self.scale_limits else 1.0
            angle = np.random.uniform(low=-self.max_angle, high=self.max_angle) if self.max_angle else 0.0
            center = np.array([(max_x + min_x) / 2, (max_y + min_y) / 2])
            rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
            input_shape = np_image.shape
            np_image = cv2.warpAffine(np_image, rot_mat, (int(np_image.shape[1]), int(np_image.shape[0])),
                                      flags=cv2.INTER_LINEAR)
            np_image = np_image.reshape(input_shape)
            landmarks = np.matmul(np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], 1), np.array(rot_mat).transpose())
            # show_image_with_keypoints(Image.fromarray(np.uint8(np_image[:, :, 0])), landmarks).show()

        return np_image, landmarks


class RandomSquash:
    def __init__(self, squash_std):
        self.squash_std = squash_std

    def __call__(self, np_image, landmarks):
        min_x, max_x, min_y, max_y = get_min_max_landmarks(landmarks)
        src_pts = np.array([(min_x, min_y), (min_x, max_y), (max_x, min_y)]).astype(np.float32)
        dst_pts = src_pts.copy()
        dst_pts[1, 0] = dst_pts[1, 0] + np.random.normal() * self.squash_std
        warp_mat = cv2.getAffineTransform(src_pts, dst_pts.astype(src_pts.dtype))
        input_shape = np_image.shape
        np_image = cv2.warpAffine(np_image, warp_mat, (np_image.shape[1], np_image.shape[0]), flags=cv2.INTER_LINEAR)
        np_image = np_image.reshape(input_shape)

        landmarks = np.matmul(np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], 1), np.array(warp_mat).transpose())
        # show_image_with_keypoints(Image.fromarray(np.uint8(np_image[:, :, 0])), landmarks).show()

        return np_image, landmarks


class CropLandmarks:

    def __init__(self, ratio=1.2, random_crop=False, return_crop_point=False):
        self.ratio = ratio
        self.random_crop = random_crop
        self.return_crop_point = return_crop_point

    def __call__(self, np_image, landmarks, inds=None):
        # select starting box points
        if inds is None:
            nonan_landmarks = landmarks[~np.isnan(landmarks[:, 0]), :]
            min_x, max_x = nonan_landmarks[:, 0].min(), nonan_landmarks[:, 0].max()
            min_y, max_y = nonan_landmarks[:, 1].min(), nonan_landmarks[:, 1].max()
        else:
            min_x, min_y, max_x, max_y = inds

        # find box size and center point
        c = (max_y + min_y) / 2, (max_x + min_x) / 2
        v = self.ratio * max(max_y - min_y, max_x - min_x)
        if self.random_crop:
            x_offset = int(np.random.uniform(low=-(v-(max_x-min_x))/2, high=(v-(max_x-min_x))/2))
            y_offset = int(np.random.uniform(low=-(v-(max_y-min_y))/2, high=(v-(max_y-min_y))/2))
            c = c[0]+y_offset, c[1]+x_offset

        crop_x = [int(c[1] - v / 2), int(c[1] + v / 2)]
        crop_y = [int(c[0] - v / 2), int(c[0] + v / 2)]

        # pad before crop
        pad_size = max([0, -crop_x[0], -crop_y[0], crop_x[1]-np_image.shape[1], crop_y[1]-np_image.shape[0]])+1
        pad_offset = np.array([pad_size, pad_size])
        padded_image = np.zeros((np_image.shape[0]+2*pad_size, np_image.shape[1]+2*pad_size, np_image.shape[2]), dtype=np.float32)
        padded_image[pad_size:pad_size + np_image.shape[0], pad_size:pad_size + np_image.shape[1], :] = np_image
        padded_landmarks = landmarks + pad_offset
        # show_image_with_keypoints(Image.fromarray(np.uint8(padded_image)), landmarks).show()

        pad_crop_x = crop_x[0]+pad_size, crop_x[1]+pad_size
        pad_crop_y = crop_y[0]+pad_size, crop_y[1]+pad_size
        crop_image = padded_image[pad_crop_y[0]:pad_crop_y[1], pad_crop_x[0]:pad_crop_x[1], :]
        crop_landmarks = padded_landmarks - np.array([pad_crop_x[0], pad_crop_y[0]])
        # show_image_with_keypoints(Image.fromarray(np.uint8((crop_image[:, :, 0]-crop_image.min())*255/(crop_image.max()-crop_image.min()))), crop_landmarks).show()

        if self.return_crop_point:
            return crop_image, crop_landmarks, (crop_x, crop_y)
        else:
            return crop_image, crop_landmarks


class Resize:

    def __init__(self, image_size=(384, 384), return_scale=False):
        if image_size[0] != image_size[1]:
            raise NotImplemented
        self.image_size = tuple(image_size)
        self.return_scale = return_scale

    def __call__(self, np_image, landmarks):
        input_shape = np_image.shape
        resized_image = cv2.resize(np_image, self.image_size, interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.reshape(self.image_size + input_shape[2:])
        scale = self.image_size[0] / np_image.shape[0]
        resized_landmarks = landmarks * scale
        # show_image_with_keypoints(Image.fromarray(np.uint8(resize_image)), landmarks).show()

        if self.return_scale:
            return resized_image, resized_landmarks, scale
        else:
            return resized_image, resized_landmarks


class CutOut:
    def __init__(self, cutout_prob=0.6, min_cutout=5, max_cutout=50):
        self.cutout_prob = cutout_prob
        self.min_cutout = min_cutout
        self.max_cutout = max_cutout

    def __call__(self, np_image, landmarks):
        if np.random.random() < self.cutout_prob:
            cut_x = np.int(np.random.uniform(low=self.min_cutout, high=self.max_cutout))
            cut_y = np.int(np.random.uniform(low=self.min_cutout, high=self.max_cutout))
            ind_x = np.int(np.random.uniform(low=0, high=np_image.shape[1] - cut_x))
            ind_y = np.int(np.random.uniform(low=0, high=np_image.shape[0] - cut_y))
            fill = np.uint8(np.random.uniform(low=0, high=256))
            np_image[ind_y:ind_y + cut_y, ind_x:ind_x + cut_x, :] = fill

        return np_image, landmarks


class Awgn:
    def __init__(self, noise_prob=0.6, snr_db=10):
        self.noise_prob = noise_prob
        self.snr = np.power(10, snr_db/10)

    def __call__(self, np_image, landmarks):
        if np.random.random() < self.noise_prob:
            s = np.linalg.norm(np_image) / np.sqrt(np_image.size)
            n = np.random.normal(scale=s/self.snr, size=np_image.shape).astype(np.float32)
            np_image = np_image + n

        return np_image, landmarks
