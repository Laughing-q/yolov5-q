# from torch.utils.data import Dataset, DataLoader
#
#
# class Test(Dataset):
#     def __init__(self) -> None:
#         super().__init__()
#
#     def __getitem__(self, index):
#         return dict(
#             box="1",
#             mask="2",
#             label="3",
#         )
#
#     def __len__(self):
#         return 10
#
#
# dataset = Test()
# test_loader = DataLoader(dataset, batch_size=4)

# from yolov5.core import MlvlPointGenerator
# import torch
#
# prior_generator = MlvlPointGenerator([8, 16, 32], offset=[0, 0])
# featmap_sizes = [[80, 80], [40, 40], [20, 20]]
# mlvl_priors = prior_generator.grid_priors(
#     featmap_sizes,
#     with_stride=True,
# )
# print(mlvl_priors[0], mlvl_priors[1])
# flatten_priors = torch.cat(mlvl_priors)
# print(flatten_priors.shape)
# flatten_priors = flatten_priors.repeat(1, 1)
# print(flatten_priors.shape)

# from yolov5.core import multi_apply
# import torch
#
# def testA(a):
#     return torch.tensor([[1, 1], [2, 2], [3, 3]]), [torch.tensor([2, 2, 2]), torch.tensor([3, 3]), torch.tensor([4])]
#
# output = multi_apply(testA, [1, 2, 3, 4])[1]
# print(*output)
# for i in zip(*output):
#     print(torch.cat(i))
# exit()
# stats = [np.concatenate(x, 0) for x in zip(*output)]  # to numpy
# output = torch.stack(output, dim=1)
# print(output, output.shape, output.dim())
# output = output.sum(0).tolist()
# print(output)

# import torch
# a = torch.zeros((2, 3), dtype=torch.bool)
# a[1, 2] = 1
# b = torch.as_tensor(a.clone(), dtype=torch.long)
# print(b)
# b[b==1] = 2
# print(b)
# # print((a == 1).sum())
# a = [torch.tensor([1, 2, 3, 1]), torch.tensor([4, 1, 6])]
# for i, pos in enumerate(a):
#     pos[pos == 1] = 50
# print(a)

# from yolov5.utils.segment import resample_segments
# import numpy as np
#
# segment = [
#     0.62548828125,
#     0.5201822916666666,
#     0.63916015625,
#     0.5045572916666666,
#     0.64892578125,
#     0.498046875,
#     0.66259765625,
#     0.4954427083333333,
#     0.67626953125,
#     0.4954427083333333,
#     0.69189453125,
#     0.5032552083333334,
#     0.70068359375,
#     0.517578125,
#     0.70556640625,
#     0.5345052083333334,
#     0.70751953125,
#     0.5631510416666666,
#     0.69775390625,
#     0.5891927083333334,
#     0.68994140625,
#     0.6022135416666666,
#     0.67822265625,
#     0.6126302083333334,
#     0.67041015625,
#     0.6178385416666666,
#     0.64990234375,
#     0.6217447916666666,
#     0.63427734375,
#     0.6204427083333334,
#     0.62451171875,
#     0.6204427083333334,
#     0.61962890625,
#     0.6165364583333334,
#     0.61572265625,
#     0.6204427083333334,
#     0.61279296875,
#     0.619140625,
#     0.61083984375,
#     0.6165364583333334,
#     0.61083984375,
#     0.6126302083333334,
#     0.61279296875,
#     0.607421875,
#     0.61474609375,
#     0.607421875,
#     0.61083984375,
#     0.599609375,
#     0.61083984375,
#     0.5748697916666666,
#     0.61376953125,
#     0.5540364583333334,
#     0.61767578125,
#     0.537109375,
#     0.62548828125,
#     0.5201822916666666,
# ]
# segment = np.array(segment).reshape(-1, 2)
# print(segment.shape)
# nsegment = resample_segments([segment])
# print(nsegment[0].shape)
# print(nsegment)

# import random
# print(random.choices(range(0), k=0))
# print(random.choice(range(10)))

import pycocotools.mask as maskUtils
import numpy as np
import cv2

def polygon_to_bitmap(polygons, height, width):
    """Convert masks from the form of polygons to bitmaps.

    Args:
        polygons (list[ndarray]): masks in polygon representation
        height (int): mask height
        width (int): mask width

    Return:
        ndarray: the converted masks in bitmap representation
    """
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)
    bitmap_mask = maskUtils.decode(rle).astype(bool)
    return bitmap_mask

polygon = [np.array([0, 0, 0, 50, 50, 50]), np.array([50, 50, 50, 100, 100, 100])]
# polygon = np.array([[0, 0, 0, 50, 50, 50], [50, 50, 50, 100, 100, 100]])
print(list(polygon))
bitmap = polygon_to_bitmap(list(polygon), 100, 100).astype(np.uint8) * 255
print(bitmap)
cv2.imshow("p", bitmap)
cv2.waitKey(0)
