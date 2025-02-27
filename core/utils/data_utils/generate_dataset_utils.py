
import datetime
import numpy as np
import os

from core.utils.data_utils.image_augmentation_utils import data_augment
from core.utils.data_utils.image_io_utils import load_image, save_to_image, save_to_image_gdal

def generate_dataset_random(image_paths, label_paths, dst_dir = "./training", image_num_per_tile=10, augmentations=[], img_h=256, img_w=256, label_is_gray=True, use_gdal=False):
    # Assuming that the source images are common images with 3 bands, and the label images are images with 1 or 3 bands.
    """
    create data sets for training and validation
    :param image_num_per_tile: sample number per tile
    :param mode: 'original' or 'augment'
    :return: None
    """
    # check source directories and create directories to store sample images and gts
    if not os.path.exists("{}/image".format(dst_dir)):
        os.mkdir("{}/image".format(dst_dir))
    if not os.path.exists("{}/label".format(dst_dir)):
        os.mkdir("{}/label".format(dst_dir))

    # number of samples for each image
    for image_path, label_path in zip(image_paths, label_paths):
        image = load_image(image_path, is_gray=False, use_gdal=use_gdal)
        label = load_image(label_path, is_gray=label_is_gray, use_gdal=use_gdal)
        image_height, image_width, _ = image.shape
        image_tag = os.path.basename(image_path).split(".")[0]
        print("%s: sampling from [%s] [%s]..." % (datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"), image_path, label_path))
        # if the source image/label is too small, pad it with zeros
        if image_height < img_h:
            image = np.pad(image, ((0, img_h-image_height+1), (0, 0), (0, 0)), mode="constant", constant_values=0)
            label = np.pad(label, ((0, img_h - image_height+1), (0, 0), (0, 0)), mode="constant", constant_values=0)
        if image_width < img_w:
            image = np.pad(image, ((0, 0), (0, img_w - image_width+1), (0, 0)), mode="constant", constant_values=0)
            label = np.pad(label, ((0, 0), (0, img_w - image_width+1), (0, 0)), mode="constant", constant_values=0)

        l_count=0
        while l_count < image_num_per_tile:
            #  randomly select a x and y for the upper left pixel
            x = np.random.randint(0, image.shape[1] - img_w+1)
            y = np.random.randint(0, image.shape[0] - img_h+1)
            src_roi = image[y:y+img_h, x:x+img_w]
            label_roi = label[y:y+img_h, x:x+img_w]
            if src_roi.shape[0] != img_h or src_roi.shape[1] != img_w:
                continue
            # data augmentation
            if len(augmentations) > 0:
                src_roi, label_roi = data_augment(src_roi, label_roi, augmentations, img_h, img_w)

            if src_roi.shape[0]!=img_h or src_roi.shape[1]!=img_w or label_roi.shape[0]!=img_h or label_roi.shape[1]!=img_w:
                continue
            if not use_gdal:
                save_to_image(src_roi.astype(np.uint8), '{}/image/{}_{}.png'.format(dst_dir, image_tag, l_count))
                save_to_image(label_roi.astype(np.uint8), '{}/label/{}_{}.png'.format(dst_dir, image_tag, l_count))
            else:
                save_to_image_gdal(src_roi, '{}/image/{}_{}.tif'.format(dst_dir, image_tag, l_count))
                save_to_image_gdal(label_roi, '{}/label/{}_{}.tif'.format(dst_dir, image_tag, l_count))
            l_count += 1


def generate_dataset_scan(image_paths, label_paths, dst_dir = "./training", stride=256, augmentations=[], img_h=256, img_w=256, label_is_gray=True, use_gdal=False):
    # Assuming that the source images are remote sensing images, and the label images are images with 1 or 3 bands.
    # check source directories and create directories to store sample images and gts
    if not os.path.exists("{}/image".format(dst_dir)):
        os.mkdir("{}/image".format(dst_dir))
    if not os.path.exists("{}/label".format(dst_dir)):
        os.mkdir("{}/label".format(dst_dir))


    for image_path, label_path in zip(image_paths, label_paths):
        image = load_image(image_path, is_gray=False, use_gdal=use_gdal)
        label = load_image(label_path, is_gray=label_is_gray, use_gdal=use_gdal)
        image_height, image_width, _ = image.shape
        image_tag = os.path.basename(image_path).split(".")[0]
        print("%s: sampling from [%s]..." % (datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"), image_path))
        # if the source image/label is too small, pad it with zeros
        if image_height < img_h:
            image = np.pad(image, ((0, img_h-image_height+1), (0, 0), (0, 0)), mode="constant", constant_values=0)
            label = np.pad(label, ((0, img_h - image_height+1), (0, 0), (0, 0)), mode="constant", constant_values=0)
        if image_width < img_w:
            image = np.pad(image, ((0, 0), (0, img_w - image_width+1), (0, 0)), mode="constant", constant_values=0)
            label = np.pad(label, ((0, 0), (0, img_w - image_width+1), (0, 0)), mode="constant", constant_values=0)

        l_count = 1
        for _row in range(0, image_height, stride):
            for _col in range(0, image_width, stride):
                src_roi = image[_row:_row+img_h, _col: _col+img_w]
                label_roi = label[_row:_row+img_h, _col:_col+img_w]
                if src_roi.shape[0]!=img_h or src_roi.shape[1]!=img_w:
                    continue

                if len(augmentations)>0:
                    src_roi, label_roi = data_augment(src_roi, label_roi, augmentations, img_h, img_w)

                if src_roi.shape[0]!=img_h or src_roi.shape[1]!=img_w or label_roi.shape[0]!=img_h or label_roi.shape[1]!=img_w:
                    continue
                # save sample images
                if not use_gdal:
                    save_to_image(src_roi.astype(np.uint8), '{}/image/{}_{}.png'.format(dst_dir, image_tag, l_count))
                    save_to_image(label_roi.astype(np.uint8), '{}/label/{}_{}.png'.format(dst_dir, image_tag, l_count))
                else:
                    save_to_image_gdal(src_roi, '{}/image/{}_{}.tif'.format(dst_dir, image_tag, l_count))
                    save_to_image_gdal(label_roi, '{}/label/{}_{}.tif'.format(dst_dir, image_tag, l_count))
                l_count += 1

def generate_dataset_main(dataset_name, args):
    image_dir = args["image_dir"]
    label_dir = args["label_dir"]
    image_paths = [os.path.join(image_dir, fn) for fn in os.listdir(image_dir)]
    label_paths = [os.path.join(label_dir, fn) for fn in os.listdir(label_dir)]
    if dataset_name=="voc" or dataset_name=="ade20k":
        image_paths = [os.path.join(image_dir, fn.replace(".png", ".jpg")) for fn in os.listdir(label_dir)]
    if args["method"] == "random":
        generate_dataset_random(image_paths=image_paths, label_paths=label_paths,
                                image_num_per_tile=args["image_number_per_tile"],
                                dst_dir=args["dst_dir"], augmentations=args["augmentations"],
                                img_h=args["image_height"], img_w=args["image_width"], use_gdal=args["use_gdal"],
                                label_is_gray=args["label_is_gray"])
    else:
        generate_dataset_scan(image_paths=image_paths, label_paths=label_paths,
                              stride=args["stride"],
                              dst_dir=args["dst_dir"], augmentations=args["augmentations"],
                              img_h=args["image_height"], img_w=args["image_width"], use_gdal=args["use_gdal"],
                              label_is_gray=args["label_is_gray"])