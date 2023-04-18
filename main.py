import os
from pathlib import Path
from typing import List

import cv2
from cv2 import dnn_superres
from loguru import logger
from PIL import Image


class ImgProcessor:
    
    def __init__(self):
        self.sr = dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel("FSRCNN_x2.pb")
        self.sr.setModel("fsrcnn", 2)

    def list_image_fps(self, dir_path: str, valid_exts: List[str] = [".png", ".jpg"]) -> List[str]:
        """
        Returns a list of file paths of images with extensions in `valid_exts` found in the directory `dir_path`.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            logger.error(f"{dir_path} is not a valid directory path.")
            return []

        all_fps = []
        for fn in os.listdir(dir_path):
            if Path(fn).suffix.lower() in valid_exts:
                all_fps.append(str(dir_path / fn))

        return all_fps

    def get_image_size_from_fp(self, img_fp: str):
        """
        Returns the width and height of the image file at path `img_fp`.
        """
        img_path = Path(img_fp)
        if not img_path.is_file():
            logger.error(f"{img_fp} is not a valid file path.")
            return {}

        with Image.open(img_path) as img:
            return {"width": img.width, "height": img.height}

    def upscale_image(self, img_input_fp: str, img_output_fp: str) -> bool:
        """
        Upscales the image file at path `img_input_fp` using the Super Resolution algorithm
        and saves the result to `img_output_fp`.
        """
        image = cv2.imread(img_input_fp)
        result = self.sr.upsample(image)
        cv2.imwrite(img_output_fp, result)
        return True

    def resize_down_image_height(self, img_input_fp: str, img_output_fp: str, img_height: int) -> bool:
        """
        Resizes the height of the image file at path `img_input_fp` to `img_height` while
        maintaining aspect ratio and saves the result to `img_output_fp`.
        """
        with Image.open(img_input_fp) as img:
            width, height = img.size
            if height < img_height:
                logger.warning(f"Input image height is too small. Current: {height}. Wanted: {img_height}")
                return False
            scaling_factor = img_height / height
            new_width = int(width * scaling_factor)
            resized_image = img.resize((new_width, img_height))
            resized_image.save(img_output_fp)
        return True

    def convert_img_to_jpg(self, img_input_fp: str, img_output_fp: str, quality: int = 90) -> bool:
        """
        Converts the image file at path `img_input_fp` from not JPG to JPG format with a specified
        quality level and saves the result to `img_output_fp`.
        """
        with Image.open(img_input_fp) as png_image:
            jpg_image = png_image.convert('RGB')
            jpg_image.save(img_output_fp, format="JPG", quality=quality)
        return True
        
    def process_supermarket_images(self, img_dir: str, min_height: int = 400) -> bool:
        """
        Iterates over `img_dir` directory, upscales (if required), downscales (if required), converts
        to JPG format (if required).
        """
        # Validate if the directory exists
        if not os.path.isdir(img_dir):
            logger.error(f"Invalid directory path: {img_dir}")
            return False

        img_fps = self.list_image_fps(dir_path=img_dir, valid_exts=[".png", ".jpg"])

        for img_fp in img_fps:
            # Get image size
            img_size = self.get_image_size_from_fp(img_fp=img_fp)
            if not img_size:
                continue

            # Check if image height is less than the minimum height, if yes then upscale the image
            current_width, current_height = img_size["width"], img_size["height"]
            if current_height < min_height:
                upscaled_img = self.upscale_image(img_input_fp=img_fp, img_output_fp=img_fp)
                if not upscaled_img:
                    logger.warning(f"Failed to upscale image: {img_fp}")
                else:
                    logger.info(f"Upscaled image: {img_fp}")

            # Get image size after upscaling if any
            img_size = self.get_image_size_from_fp(img_fp=img_fp)
            if not img_size:
                continue

            # Check if image height is greater than the minimum height, if yes then resize the image
            current_width, current_height = img_size["width"], img_size["height"]
            if current_height > min_height:
                resized_down = self.resize_down_image_height(img_input_fp=img_fp, img_output_fp=img_fp, img_height=min_height)
                if not resized_down:
                    logger.warning(f"Failed to resize image: {img_fp}")
                else:
                    logger.info(f"Resized image: {img_fp}")

            # Check if image is in PNG format, if yes then convert it to JPG
            if Path(img_fp).suffix in [".png", ".jpeg"]:
                converted = self.convert_img_to_jpg(img_input_fp=img_fp, img_output_fp=img_fp)
                if not converted:
                    logger.warning(f"Failed to convert image: {img_fp}")
                else:
                    logger.info(f"Converted image: {img_fp}")

        return True