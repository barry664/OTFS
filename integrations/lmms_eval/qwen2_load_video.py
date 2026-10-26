from typing import Optional, Union, Tuple, List, Any, Dict
import time
import torch
import decord
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import logging

# Import dependencies required by the updated qwen_vl_utils API.
from qwen_vl_utils.vision_process import (
    smart_resize, 
    fetch_image, 
    extract_vision_info,
    SPATIAL_MERGE_SIZE,
    VIDEO_MIN_TOKEN_NUM,
    VIDEO_MAX_TOKEN_NUM,
    MODEL_SEQ_LEN,
    FRAME_FACTOR
)

logger = logging.getLogger(__name__)

def _read_video_decord_frame_idx(
    ele: Dict[str, Any], frame_idx: list, nframes: int
) -> Tuple[torch.Tensor, dict, float]:
    """read video using decord.VideoReader with specific frame indices."""
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)

    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")

    total_frames, video_fps = len(vr), vr.get_avg_fps()
    
    # Decode the specified frames.
    idx = frame_idx[:nframes]
    idx = sorted(idx)
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    
    logger.info(f"decord_frame_idx: {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    # Estimate the sampling frame rate for downstream processing.
    sample_fps = len(idx) / max(total_frames, 1e-6) * video_fps

    # Build the video metadata required by the updated API.
    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="decord",
    )

    return video, video_metadata, sample_fps


from torchvision import transforms
from torchvision.transforms import InterpolationMode
from typing import Union
from PIL import Image

def fetch_video_frame_idx(
    ele: Dict[str, Any], 
    frame_idx: list, 
    nframes: int, 
    image_patch_size: int = 14,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False
) -> Union[torch.Tensor, List[Image.Image], Tuple]:
    
    # Compute resize factors for the current API.
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    VIDEO_FRAME_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    VIDEO_FRAME_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor

    # Read the selected frames with the custom Decord loader.
    video, video_metadata, sample_fps = _read_video_decord_frame_idx(ele, frame_idx, nframes)
    
    nframes_actual, _, height, width = video.shape

    # Apply per-frame pixel limits consistent with the updated API.
    min_pixels = ele.get("min_pixels", VIDEO_FRAME_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9)
    max_pixels = max(min(VIDEO_FRAME_MAX_PIXELS, total_pixels / nframes_actual * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    
    if max_pixels_supposed > max_pixels:
        logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
    max_pixels = min(max_pixels_supposed, max_pixels)

    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    # Return video metadata when requested.
    final_video = (video, video_metadata) if return_video_metadata else video
    if return_video_sample_fps:
        return final_video, sample_fps
    
    return final_video


from typing import Optional

def process_vision_info_frame_idx(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    frame_idx: list = None,
    nframes: int = None,
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    image_patch_size: int = 14,
) -> Tuple[Optional[List[Image.Image]], Optional[List[Union[torch.Tensor, List[Image.Image]]]], Optional[Dict[str, Any]]]:
    
    vision_infos = extract_vision_info(conversations)

    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            # Load images with the standard fetch_image function.
            image_inputs.append(fetch_image(vision_info, image_patch_size=image_patch_size))
        elif "video" in vision_info:
            # Load videos with the custom frame-index loader.
            video_input, video_sample_fps = fetch_video_frame_idx(
                vision_info, 
                frame_idx=frame_idx, 
                nframes=nframes,
                return_video_sample_fps=True,
                image_patch_size=image_patch_size,
                return_video_metadata=return_video_metadata
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")

    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None

    # Pass video sampling arguments compatible with Qwen2.5-VL.
    video_kwargs = {'do_sample_frames': False}
    if not return_video_metadata: # BC for qwen2.5vl
        video_kwargs.update({'fps': video_sample_fps_list})

    if return_video_kwargs:
        return image_inputs, video_inputs, video_kwargs
        
    return image_inputs, video_inputs
