#!/opt/conda/bin/python
#from mmsegmentation.build.lib.mmseg.core import seg
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

EPS = 1e-2


def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


def class_names(label_path):
    """Maps the lable to class number from tet file

    Args:
        label_path: Path to the class_names

    Returns:
        class_list: A list of class names in order.
    """

    classes = open(label_path, "r")

    class_list = [i.strip() for i in classes.readlines()]
    return class_list


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      detection_name="person",
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=13,
                      win_name='',
                      show=False,
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.
    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray or None): Masks, shaped (n,h,w) or None
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.  Default: 0
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        show (bool): Whether to show the image. Default: True
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None
    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    mask_colors = []
    if labels.shape[0] > 0:
        if mask_color is None:
            # Get random state before set seed, and restore random state later.
            # Prevent loss of randomness.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            # random color
            np.random.seed(42)
            mask_colors = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            np.random.set_state(state)
        else:
            # specify  color
            mask_colors = [
                np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
            ] * (
                max(labels) + 1)

    bbox_color = color_val_matplotlib(bbox_color)
    text_color = color_val_matplotlib(text_color)

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    polygons = []
    color = []
    arr_bbox = BoundingBoxes()
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        if (label_text != detection_name):
            continue
        bb = BoundingBox()
        bb.probability = bbox[-1]
        bb.xmin = int(bbox[0].item())
        bb.ymin = int(bbox[1].item())
        bb.xmax = int(bbox[2].item())
        bb.ymax = int(bbox[3].item())
        bb.Class = label_text
        arr_bbox.bounding_boxes.append(bb)

        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(bbox_color)
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        ax.text(
            bbox_int[0],
            bbox_int[1],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=font_size,
            verticalalignment='top',
            horizontalalignment='left')
        if segms is not None:
            color_mask = mask_colors[labels[i]]
            mask = segms[i].astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    if show:
        plt.imshow(img)

    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    ax.add_collection(p)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)
    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img, arr_bbox


def show_result(
        img,
        result,
        label_path,
        detection_name,
        score_thr=0.3,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
        mask_color=None,
        thickness=2,
        font_size=13,
        win_name='',
        show=False,
        wait_time=0,
        out_file=None):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor or tuple): The results to draw over `img`
            bbox_result or (bbox_result, segm_result).
        score_thr (float, optional): Minimum score of bboxes to be shown.
            Default: 0.3.
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
            The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
            The tuple of color should be in BGR order. Default: 'green'
        mask_color (None or str or tuple(int) or :obj:`Color`):
            Color of masks. The tuple of color should be in BGR order.
            Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)

    # segm_person_image = None
    # if segms is not None:
    #     if segms[0] is not None:
    #         # print(segm_result[0])
    #         segm_result_np = np.where(segms == 1, 255,segms )
    #         # print(segm_result_np.shape)
    #         segm_person_tensor = torch.tensor(segm_result_np,dtype=torch.float32)

    #         #segm_person_tensor = torch.stack(segm_person_tensor, dim=0)
    #         print("----------------------------------------------3")
    #         print(segm_person_tensor.shape)
    #         segm_person_tensor = segm_person_tensor.softmax(1)
    #         segm_person_tensor = segm_person_tensor[0]
    #         print(segm_person_tensor)
    #         #segm_person_tensor = segm_person_tensor.view(segm_person_tensor.shape[1], segm_person_tensor.shape[2], segm_person_tensor.shape[0])
    #         segm_person_image = segm_person_tensor.detach().cpu().numpy()
    #         #segm_person_image = segm_person_image[0]
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    img, arr_bbox = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        detection_name,
        segms,
        class_names=class_names(label_path),
        score_thr=score_thr,
        bbox_color=bbox_color,
        text_color=text_color,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)

    if not (show or out_file):
        return img, arr_bbox
