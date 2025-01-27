from pathlib import Path

import cv2
import data.util as data_util
import models.archs.TCVC_IDC_arch as TCVC_IDC_arch
import numpy as np
import torch
import torch.nn.functional as F
import utils.util as util


def save_images(start_idx, end_idx, output_dir, images, image_paths):
    """
    Save a list of images to the output directory.
    Args:
        start_idx (int): Starting index of the images.
        end_idx (int): Ending index of the images.
        output_dir (str): Directory to save images.
        images (list): List of images to save.
        image_paths (list): List of corresponding image paths.
    """
    for idx, image in enumerate(images):
        image_name = Path(image_paths[idx]).name
        save_path = Path(output_dir) / image_name
        cv2.imwrite(str(save_path), image[:, :, ::-1])


def create_directory(path):
    """
    Create a directory if it doesn't exist.
    Args:
        path (str or Path): Path to the directory.
    """
    directory = Path(path)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


class TCVC:
    def __init__(self, input_folder, output_folder, model_path):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.model_path = model_path

        create_directory(self.output_folder)

        self.model = self.initialize_model()

    def initialize_model(self):
        """
        Initialize the TCVC model.
        Returns:
            model (torch.nn.Module): Initialized model.
        """
        model = TCVC_IDC_arch.TCVC_IDC(
            nf=64, N_RBs=3, key_net="sig17", dataset="DAVIS4"
        )
        model.load_state_dict(torch.load(self.model_path), strict=True)
        model.eval()
        return model.cuda()

    def process_video(
        self, video_folder, save_subfolder, interval_length=17, image_size=256
    ):
        """
        Process a single video folder.
        Args:
            video_folder (Path): Path to the video folder.
            save_subfolder (Path): Path to save processed images.
            interval_length (int): Interval length for keyframes.
            image_size (int): Image resizing size.
        """
        supported_extensions = ("*.jpeg", "*.jpg", "*.png", "*.bmp")
        image_paths = sorted(
            [img for ext in supported_extensions for img in video_folder.glob(ext)]
        )

        if not image_paths:
            print(f"No valid images found in {video_folder}")
            return

        images = [data_util.read_img(None, str(img)) / 255.0 for img in image_paths]
        rgb_flag = images[0].shape[-1] == 3

        keyframe_indices = list(range(0, len(images), interval_length + 1))
        if keyframe_indices[-1] == len(images) - 1:
            keyframe_indices.pop()

        print(f"Processing video: {video_folder.name}")
        print(f"Total images: {len(images)}, Keyframes: {keyframe_indices}")

        for k in keyframe_indices:
            img_subset_paths = image_paths[k : k + interval_length + 2]
            img_subset = np.stack(images[k : k + interval_length + 2], axis=0)

            # Reorders the img_in dimensions as (N, C, H, W)
            # This reordering is necessary because PyTorch expects
            # image tensors in the shape (N, C, H, W) where C is the number of channels.
            # Converts reordered NumPy array into PyTorch tensors and then converts it into float
            img_tensor = torch.from_numpy(img_subset.transpose(0, 3, 1, 2)).float()
            img_l_tensor = self.prepare_input(img_tensor, rgb_flag, image_size)

            with torch.no_grad():
                out_ab, *_ = self.model(img_l_tensor)

            out_rgb_images = self.post_process_output(img_tensor, out_ab)
            save_images(
                k,
                k + len(out_rgb_images),
                save_subfolder,
                out_rgb_images,
                img_subset_paths,
            )

    def prepare_input(self, img_tensor, rgb_flag, image_size):
        """
        Prepare the input tensor for the model.
        Args:
            img_tensor (torch.Tensor): Input image tensor.
            rgb_flag (bool): Whether the input is RGB.
            image_size (int): Resizing size.
        Returns:
            img_l_tensor (torch.Tensor): Processed input tensor.
        """
        if rgb_flag:
            img_lab_tensor = data_util.rgb2lab(img_tensor)
            img_l_tensor = img_lab_tensor[:, :1, :, :]
        else:
            img_l_tensor = img_tensor - 0.5

        img_l_resized = F.interpolate(
            img_l_tensor, size=(image_size, image_size), mode="bilinear"
        )
        return [
            img_l_resized[i : i + 1, ...].cuda() for i in range(img_l_resized.shape[0])
        ]

    def post_process_output(self, img_tensor, out_ab):
        """
        Post-process the output from the model.
        Args:
            img_tensor (torch.Tensor): Original image tensor.
            out_ab (torch.Tensor): Model output tensor.
        Returns:
            list: List of RGB images.
        """
        out_ab = out_ab.detach().cpu()[0, ...]
        _, _, H, W = img_tensor.size()

        out_a_resized = F.interpolate(out_ab[:, :1, :, :], size=(H, W), mode="bilinear")
        out_b_resized = F.interpolate(
            out_ab[:, 1:2, :, :], size=(H, W), mode="bilinear"
        )

        out_lab = torch.cat((img_tensor, out_a_resized, out_b_resized), dim=1)
        out_rgb = data_util.lab2rgb(out_lab)

        return [
            util.tensor2img(out_rgb[i].clip(0, 1) * 255, np.uint8)  # Keep as tensor
            for i in range(out_rgb.size(0))
        ]

    def colorize(self):
        """
        Main function to process all videos in the input folder.
        """
        video_folders = [
            folder for folder in self.input_folder.iterdir() if folder.is_dir()
        ]

        for video_folder in video_folders:
            save_subfolder = self.output_folder / video_folder.name
            create_directory(save_subfolder)
            self.process_video(video_folder, save_subfolder)
