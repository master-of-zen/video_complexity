#!/usr/bin/env python

import itertools
import sys
import time
import cv2
import numpy as np
import scipy.ndimage
from matplotlib import pyplot

def image_complexity_jpeg(image):
    r, buf = cv2.imencode('.jpeg', image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return len(buf)


def image_complexity_webp(image):
    r, buf = cv2.imencode(".webp", image, [cv2.IMWRITE_WEBP_QUALITY, 100])
    return len(buf)


def image_complexity_iterated(filter, levels=None):
    def f(image):
        image_nx, image_ny, img_nchannels = image.shape
        image_small = np.copy(image)
        level = 0
        total = 0

        while True:
            image_small_nx, image_small_ny, image_small_nchannels = image_small.shape
            if image_small_nx == 1 or image_small_ny == 1 or (levels != None and level > levels):
                break

            image_small = cv2.resize(image_small, (image_small_ny // 2, image_small_nx // 2),
                                     interpolation=cv2.INTER_LINEAR)

            total += filter(image_small)
            level += 1

        return total

    return f


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
    """Converts an image filter into a video filter by frame"""

    def f(iterator):
        for i, frame in iterator:
            yield i, image_filter(frame)

    return f


# --- media loading ---

def is_image(path):
    """OpenCV supported image formats"""
    extensions = [
        'bmp', 'dib',
        'jpeg', 'jpg', 'jpe',
        'jp2',
        'png',
        'webp'
        'pbm', 'pgm', 'ppm', 'pxm', 'pnm',
        'pfm',
        'sr', 'ras',
        'tiff', 'tif',
        'exr',
        'hdr', 'pic',
    ]

    return any(path.lower().endswith('.' + i) for i in extensions)


def load_video(filename):
    video = cv2.VideoCapture(filename)
    for i in itertools.count():
        has_frame, frame = video.read()
        if not has_frame:
            break
        yield i, np.transpose(frame, (1, 0, 2)) / 255


def load_image(filename):
    return np.transpose(cv2.imread(filename), (1, 0, 2)) / 255


def get_resolution(filename):
    if is_image(filename):
        return load_image(filename).shape[:2]
    for i, f in load_video(filename):
        return f.shape[:2]


# --- main ---

methods = {
    "image:jpeg": image_complexity_jpeg,
    "image:webp": image_complexity_webp,
    "video:box": video_complexity_box(2),
    "video:gaussian": video_complexity_gaussian(radius=2, sigma=2 * 0.3)
}


def run(paths, methods=None, verbose=False):
    if methods == None:
        methods = sorted(sys.modules[__name__].methods)

    Row = make_row_type(['path', 'frame_index', 'resolution'] + methods)

    for path in paths:
        results = {}
        for method_name in methods:
            t = time.time()
            if verbose:
                print("Running", path, method_name, "...")

            method = sys.modules[__name__].methods[method_name]
            if is_image(path) and method_name.startswith('image:'):
                img = load_image(path)
                results.setdefault(None, dict())[method_name] = method(img)
            else:
                video = load_video(path)
                if method_name.startswith('image:'):
                    method = image_filter_to_video_filter(method)
                for i, val in method(video):
                    results.setdefault(i, dict())[method_name] = val
            print('Done ', method_name, f"{round(time.time() - t, 1)}")
        resolution = get_resolution(path)
        resolution_string = 'x'.join(str(d) for d in resolution)
        for i in sorted(results):
            yield Row(path=path, frame_index=i, resolution=resolution_string, **results[i])


def make_row_type(cols):
    class Row(tuple):
        columns = cols
        index = {column: cols.index(column) for column in cols}

        def __new__(_cls, *args, **kwargs):
            if len(args) == 0:
                t = (kwargs[c] if c in kwargs else None for c in cols)
                return tuple.__new__(_cls, t)
            assert len(args) == len(cols)
            return tuple.__new__(_cls, args)

        def get(self, column):
            if column not in self.index:
                raise AttributeError
            return self[self.index[column]]

        def map(self, f):
            return Row(*map(f, cols, self))

    return Row


def write_csv(iterator, fd):
    for i, row in enumerate(iterator):
        r = []
        for x in row:
            if x:
                if isinstance(x, float) or isinstance(x, int):
                    r.append(str(round(x, 2)))
                else:
                    r.append(x)
        row = r

        fd.write(','.join([str(v) if v != None else '' for v in row]) + '\n')


def read_csv(fd):
    Row = None
    for line in fd:
        fields = line.strip().split(',')
        if Row == None:
            Row = make_row_type(fields)
        else:
            fields_typed = []
            for column, value in zip(Row.columns, fields):
                if value == '':
                    fields_typed.append(None)
                elif column == 'frame_index':
                    fields_typed.append(int(value))
                elif column.startswith('video:') or column.startswith('image:'):
                    fields_typed.append(float(value))
                else:
                    fields_typed.append(value)
            yield Row(*fields_typed)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("media", nargs='+', help="the media files to analyze")
    parser.add_argument("-m", "--methods",
                        help="the methods to use on the specified files, comma separated without spaces. if not specified, all available methods are used")
    parser.add_argument("-v", "--verbose", help="verbose", action="store_true")
    parser.add_argument("-o", "--output", help="write output to file")

    args = parser.parse_args()

    m = sorted(methods)
    if args.methods != None:
        m = args.methods.split(',')

    gen = run(args.media, methods=m, verbose=args.verbose)

    if args.output != None:
        with open(args.output, 'w') as f:
            write_csv(gen, f)
    else:
        write_csv(gen, sys.stdout)

    pyplot.plot()
    pyplot.