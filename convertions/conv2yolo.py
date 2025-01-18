import os
import sys
import cv2
import numpy as np
import argparse
import json
import shutil
import random

from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm


class STOY:

    def __init__(self, source_dir, dest_dir, class_names, checker):
        self.source = source_dir
        self.dest = dest_dir

        self.checker = checker

        if self.checker:
            if os.path.exists("checking"):
                shutil.rmtree("checking")
            os.makedirs("checking")

        self.source_list = []

        cls = open(class_names, "r")
        CLASSES = cls.readlines()
        CLASSES = [_.split('\n')[0] for _ in CLASSES]
        self.class_to_ind = dict(zip(CLASSES, range(len(CLASSES))))



    def find_source_list(self):
        source_dirs = os.listdir(self.source)
        source_list = np.array([], dtype = str)

        for source_dir in source_dirs:
            if len(source_dir.split('.')) == 1:
                source_list = np.append(source_list, str(source_dir))

        for dir in source_list:
            dir = os.path.join(self.source, dir, "ann")
            if not os.path.exists(dir):
                source_list = np.delete(source_list,
                            np.where(source_list == dir.split('/')[-2]))

                print(f"\nCan not find '{dir}' directory.\n")
                continue

        self.source_list = \
            [os.path.join(self.source, _, "ann") for _ in source_list]



    def conv2yolo(self, json, txt_name, label_path):
        lab_p = os.path.join(label_path, txt_name.split('.json')[0])
        # Open the txt file replacing .jpg or .png to .txt
        txt_file = open(lab_p.replace(".jpg",".txt").replace(".png",".txt"), 'w')

        w = json['size']['width']
        h = json['size']['height']

        if not json['objects']:
            print("Error: " + lab_p.replace(".jpg",".txt").replace(".png",".txt"), "is empty\n")
            # Leave the empty file
            txt_file.close()

        if self.checker:
            img_file = lab_p.replace(self.dest, self.source)
            img_file = img_file.replace('labels', 'img')
            img = cv2.imread(img_file)

        for object in json['objects']:
            cls = self.class_to_ind[object['classTitle']]

            #(left, top), (right, bottom)
            (x1, y1), (x2, y2) = object['points']['exterior']
            x_cnt = ((x1 + x2) / 2) / w
            y_cnt = ((y1 + y2) / 2) / h
            width = abs((x2 - x1) / w)
            height = abs((y2 - y1) / h)

            width = 0 if width < 0 else width
            height = 0 if height < 0 else height
            width = w if width > w else width
            height = h if height > h else height

            txt_file.write(
                f"{cls} {x_cnt:.5f} {y_cnt:.5f} {width:.5f} {height:.5f}\n"
            )
            if self.checker:
                self.plot_one_box_ko([x1, y1, x2, y2], img, label = str(cls))

        if self.checker:
            cv2.imwrite(
                f"checking/{lab_p.split('/')[1]}/{lab_p.split('/')[-1]}",img)

        txt_file.close()
        return lab_p


    def plot_one_box_ko(self, x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

            # original code
            #cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2),cv2.FONT_HERSHEY_COMPLEX_SMALL, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

            # cv2 convert to pil and pil convert to cv2 for print korea character
            imgp = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            imgd = ImageDraw.Draw(imgp)
            color = tuple(color) # change color's type
            #imgd.rectangle([(c1[0],c1[1]), (c2[0]-40,c2[1]+10)], fill=color)
            imgd.text((c1[0],c1[1]-10),label,color)
            img = cv2.cvtColor(np.array(imgp),cv2.COLOR_RGB2BGR)

        return img


    def run(self):
        self.find_source_list()

        if os.path.exists(self.dest):
            shutil.rmtree(self.dest)
        os.makedirs(self.dest)

        print(f"\n'{self.dest}' folder is created.")

        all_path = open(os.path.join(self.dest, "train.txt"), 'w')

        for dir in self.source_list:
            dir_path = os.path.join(self.dest, dir.split('/')[-2])
            label_path = os.path.join(dir_path, 'labels')
            images_path = label_path.replace('labels', 'images')

            os.makedirs(dir_path)
            os.makedirs(label_path)
            os.makedirs(images_path)

            if self.checker:
                os.makedirs(f"checking/{dir_path.split('/')[-1]}")
                os.system(f"cp {dir.replace('ann', 'img')}/*.* {images_path}")

            for json_name in tqdm(os.listdir(dir)):
                with open(f"{dir}/{json_name}","r") as source_json:

                    source_json = json.load(source_json)
                    lab_p = self.conv2yolo(source_json, json_name, label_path)

                all_path.write(lab_p.replace("labels", "images") + '\n')

        all_path.close()

        os.system(f"chmod 777 -R {self.dest}")
        if self.checker:
            os.system(f"chmod 777 -R checking")
        print("\nAnnotation convert was done successfully.\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_dir", default = None, type = str,
            help = "source(project) directory")

    parser.add_argument("--dest_dir", default = "convert", type = str,
            help = "destination directory")

    parser.add_argument("--name", default = "class.names", type = str,
            help = "class name")

    parser.add_argument("--checker", action = 'store_true',
            help = "checking the result of anntation through images")

    args = parser.parse_args()

    if args.source_dir is None:
        print("\nError: Folder is not exists.")
        print("Please check your source_dir.\n")
        sys.exit()

    stoy = STOY(args.source_dir, args.dest_dir, args.name, args.checker)

    stoy.run()
