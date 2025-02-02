import os
import shutil
from PIL import Image

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tiff",
    ]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def class2idx(dir, idxkeys=False):
    classes = sorted(os.listdir(dir))
    if idxkeys: 
        return {idx: cls_name for idx, cls_name in enumerate(classes)}
    else:
        return {cls_name: idx for idx, cls_name in enumerate(classes)}

def make_dataset(dir):
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    images = []
    labels = []
    classes = sorted(os.listdir(dir))
    class_to_idx = class2idx(dir)

    for cls_name in classes:
        cls_dir = os.path.join(dir, cls_name)
        if not os.path.isdir(cls_dir):
            AssertionError("%s is not a valid directory" % cls_dir)
        for root, _, fnames in sorted(os.walk(cls_dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
                    labels.append(class_to_idx[cls_name])
    return images, labels


def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = (var + 1) / 2
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype("uint8"))

def del_files(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        os.makedirs(directory_path)