from gc import callbacks

import numpy as np
import torch
from cv2 import cvtColor
from torch import randint, tensor
import matplotlib.pyplot as plt
import cv2
import torchmetrics
from segment_anything import sam_model_registry, SamPredictor
import os

from torchmetrics.classification import BinaryJaccardIndex

box_points = []
box_mode_active = False

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())
        
        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.savefig(filename +'.png',bbox_inches='tight',pad_inches=-0.1)
    plt.close()

def mouse_callback(event, x, y, flags, param):
    global box_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == 'box':
            box_points.append([x, y])
            if len(box_points) == 2:
                point0 = np.array(box_points[0])
                point1 = np.array(box_points[1])
                x0, y0 = min(point0[0], point1[0]), min(point0[1], point1[1])
                x1, y1 = max(point0[0], point1[0]), max(point0[1], point1[1])
                input_box = np.array([[x0, y0, x1, y1]])

                process_box(input_box)
                final_img_path = result_path + '\\' + filename + '_0.png'
                final_img = cv2.imread(final_img_path)
                iou(input_box, final_img_path)
                if final_img is not None:
                    img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                    cv2.imshow("Final Image", img)
                else:
                    print(f"Failed to load image from {final_img_path}")
        elif mode == 'point':
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(box_points) < 3:
                    box_points.append([x, y])
                    if len(box_points) == 3:
                        input_point = np.array(box_points)
                        process_points(input_point)

                        final_img_path = result_path + '\\' + filename + '_0.png'
                        final_img = cv2.imread(final_img_path)
                        if final_img is not None:
                            img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                            cv2.imshow("Final Image", img)
                        else:
                            print(f"Failed to load image from {final_img_path}")
        elif mode == 'multibox':
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(box_points) < 4:
                    box_points.append([x, y])
                    if len(box_points) == 4:
                        point0 = np.array(box_points[0])
                        point1 = np.array(box_points[1])
                        x0, y0 = min(point0[0], point1[0]), min(point0[1], point1[1])
                        x1, y1 = max(point0[0], point1[0]), max(point0[1], point1[1])
                        point2 = np.array(box_points[2])
                        point3 = np.array(box_points[3])
                        x2, y2 = min(point2[0], point3[0]), min(point2[1], point3[1])
                        x3, y3 = max(point2[0], point3[0]), max(point2[1], point3[1])
                        input_box = np.array([[x0, y0, x1, y1],[x2, y2, x3, y3]])

                        final_img_path = result_path + '\\' + filename + '.png'
                        final_img = cv2.imread(final_img_path)

                        process_multibox(input_box, final_img_path)

                        if final_img is not None:
                            img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                            cv2.imshow("Final Image", img)
                        else:
                            print(f"Failed to load image from {final_img_path}")


def process_box(input_box):
    #msk = cv2.imread("C:\\Users\\BiancaB\\sam-hq\\demo\\input_imgs\\example5_mask.png", cv2.IMREAD_GRAYSCALE)
    #ret, binary_mask = cv2.threshold(msk, 127, 1, cv2.THRESH_BINARY)  # all pixels over 127 are 1, rest are 0

    input_point, input_label = None, None

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        #mask_input=torch.tensor(binary_mask, dtype=torch.float).unsqueeze(0),
        multimask_output=False,
        hq_token_only=hq_token_only,
    )

    os.makedirs(result_path, exist_ok=True)
    show_res(masks, scores, input_point, input_label, input_box, result_path + '\\' + filename, image)

def process_points(input_point):
    input_label = np.ones(input_point.shape[0])

    input_box = None

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box = input_box,
        multimask_output=False,
        hq_token_only=hq_token_only,
    )

    os.makedirs(result_path, exist_ok=True)
    show_res(masks,scores,input_point, input_label, input_box, result_path + '\\' + filename, image)

def process_multibox(input_box):
    input_box = torch.tensor(input_box, device=predictor.device)
    transformed_box = predictor.transform.apply_boxes_torch(input_box, image.shape[:2])
    input_point, input_label = None, None

    masks, scores, logits = predictor.predict_torch(
        point_coords=input_point,
        point_labels=input_label,
        boxes=transformed_box,
        multimask_output=False,
        hq_token_only=hq_token_only,
    )

    masks = masks.squeeze(1).cpu().numpy()
    scores = scores.squeeze(1).cpu().numpy()
    input_box = input_box.cpu().numpy()
    os.makedirs(result_path, exist_ok=True)
    show_res_multi(masks, scores, input_point, input_label, input_box, result_path + '\\' + filename, image)

def iou(input_box, img_path):
    msk = cv2.imread("C:\\Users\\BiancaB\\sam-hq\\demo\\input_imgs\\example5_mask.png", cv2.IMREAD_GRAYSCALE)
    ret, binary_mask = cv2.threshold(msk, 127, 1, cv2.THRESH_BINARY) # all pixels over 127 are 1, rest are 0

    binary_mask = torch.tensor(binary_mask, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
    input_box = torch.tensor(input_box, dtype=torch.float).to(device)
    print(input_box)

    masks, iou_predictions, low_res_masks = predictor.predict_torch(point_coords=None, point_labels=None,
                                                                    boxes=input_box, mask_input=binary_mask,
                                                                    multimask_output=False, hq_token_only=hq_token_only)

    predicted = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    resized_predicted = cv2.resize(predicted, msk.shape)

    cv2.imshow("Predicted", resized_predicted)
    ret, resized_predicted = cv2.threshold(resized_predicted, 127, 1, cv2.THRESH_BINARY) # all pixels over 127 are 1, rest are 0

    predicted_mask = torch.tensor(resized_predicted, dtype=torch.uint8)
    torch.resize_as_(predicted_mask, binary_mask)


    metric = BinaryJaccardIndex()

    iou_value = metric(predicted_mask.flatten(), binary_mask.flatten()) # from 2D to 1D - simplifies the problem
    print(iou_value, "<- IoU value")


if __name__ == "__main__":
    sam_checkpoint = "C:\\Users\\BiancaB\\sam-hq\\pretrained_checkpoint\\sam_hq_vit_tiny.pth"
    model_type = "vit_tiny"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)

    image = cv2.imread("C:\\Users\\BiancaB\\sam-hq\\demo\\input_imgs\\example5.jpg")
    result_path = 'C:\\Users\\BiancaB\\sam-hq\\demo\\hq_sam_tiny_result'
    filename = 'example5'
    predictor.set_image(image)

    hq_token_only = False

    cv2.namedWindow('image')

    while True:
        cv2.imshow('image', image)
        key=cv2.waitKey(1) & 0xFF # capture the keyboard input

        if key == ord('1'):
            print("Box mode activated. Click two points to define the box.")
            mode = 'box'
            box_points = []
            cv2.setMouseCallback('image', mouse_callback)
            while len(box_points) < 2:
                cv2.waitKey(1)  # Wait for mouse events

        if key == ord('2'):
            print("Point prompt mode activated. Click three points to define the area.")
            hq_token_only = True
            mode = 'point'
            box_points = []
            cv2.setMouseCallback('image', mouse_callback)
            while len(box_points) < 3:
                cv2.waitKey(1)

        if key == ord('3'):
            print("Multibox mode activated. Click 4 points to define the boxes.")
            mode = 'multibox'
            box_points = []
            cv2.setMouseCallback('image', mouse_callback)
            while len(box_points) < 4:
                cv2.waitKey(1)

        elif key == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()


