#!/usr/bin/env python

import json
import sys
from matplotlib import pyplot
import numpy as np


def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


file = sys.argv[1]

data = read_json(file)
jpeg = [v.get('image:jpeg') for k, v in data.items()]
jpeg_mean = np.nanmean(jpeg)
webp = [v.get('image:webp') for k, v in data.items()]
webp_mean = np.mean(webp)
video_box = [v.get('video:box') for k, v in data.items()]
video_gauss = [v.get('video:gaussian') for k, v in data.items()]

print(f'Jpeg {int(jpeg_mean)}; Webp: {int(webp_mean)}; ')
      #f'Box: {round(int(np.nanmean(video_box)))}; Video_gauss: {int(np.nanmean(video_gauss))}')
