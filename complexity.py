#!/usr/bin/env python

import itertools
import sys
import time
import cv2
import numpy as np
import scipy.ndimage
from matplotlib import pyplot
import json


def image_complexity_jpeg(image):
    r, buf = cv2.imencode('.jpeg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return len(buf)


def image_complexity_webp(image):
    r, buf = cv2.imencode(".webp", image, [cv2.IMWRITE_WEBP_QUALITY, 90])
    return len(buf)


def video_complexity_with_kernel(kernel):
    """Runs a separable kernel on all three dimensions of a video"""

    def f(frame_iterator):
        radius = (len(kernel) - 1) // 2
        frame_window_convolved = []
        frame_window_originals = []

        for i, frame in frame_iterator:
            frame_window_originals.append(frame)
            frame = scipy.ndimage.filters.convolve1d(frame, kernel, 0)
            frame = scipy.ndimage.filters.convolve1d(frame, kernel, 1)
            frame_window_convolved.append(frame)

            frame_window_originals = frame_window_originals[-len(kernel):]
            frame_window_convolved = frame_window_convolved[-len(kernel):]

            if len(frame_window_originals) == len(kernel):
                window_arr = np.array(frame_window_convolved)
                convolved = np.einsum('i...,i->...', window_arr, kernel)
                result = np.sum(np.abs(convolved - frame_window_originals[radius]))

                yield i - radius, result

    return f


def video_complexity_box(radius):
    kernel = np.ones(radius * 2 + 1) / (radius * 2 + 1)
    return video_complexity_with_kernel(kernel)


def video_complexity_gaussian(sigma, radius):
    kernel = cv2.getGaussianKernel(radius * 2 + 1, sigma).T[0]
    return video_complexity_with_kernel(kernel)


def image_filter_to_video_filter(image_filter):
    def f(iterator):
        for i, frame in iterator:
            yield i, image_filter(frame)
    return f


def load_video(filename, skip):
    video = cv2.VideoCapture(filename)
    for i in itertools.count():
        has_frame, frame = video.read()
        if not has_frame:
            break
        if i % skip == 0:
            yield i, np.transpose(frame, (1, 0, 2)) / 255


def load_image(filename):
    return np.transpose(cv2.imread(filename), (1, 0, 2)) / 255


def run(paths, skip):
    methods = {
        "image:jpeg": image_complexity_jpeg,
        "image:webp": image_complexity_webp,
        # "video:box": video_complexity_box(2),
        # "video:gaussian": video_complexity_gaussian(radius=2, sigma=2 * 0.3)
    }

    for path in paths:
        results = {}
        for method_name in methods:
            t = time.time()
            print("Running", path, method_name, "...")
            method = methods.get(method_name)

            video = load_video(path, skip)
            if method_name.startswith('image:'):
                method = image_filter_to_video_filter(method)

            for i, val in method(video):
                results.setdefault(i, dict())[method_name] = val
            print('Done ', method_name, f"{round(time.time() - t, 1)}")

        with open(f'{path}.json', 'w') as file:
            json.dump(results, file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("media", nargs='+', help="the media files to analyze")
    parser.add_argument("-o", "--output", help="write output to file")
    parser.add_argument("-s", type=int, default=1,  help="skip frames")

    args = parser.parse_args()
    run(args.media, args.s)
