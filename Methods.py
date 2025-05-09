import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import torch
import matplotlib

matplotlib.use('TkAgg')

def select_points(img, title="Select points"):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert for matplotlib
    plt.imshow(img_rgb)
    plt.title(title)
    points = plt.ginput(n=-1, timeout=0)
    plt.close()
    return np.array(points)

def add_boundary_points(img_shape):
    h, w = img_shape[:2]
    return np.array([
        [0, 0], [w-1, 0], [w-1, h-1], [0, h-1],
        [w//2, 0], [w-1, h//2], [w//2, h-1], [0, h//2]
    ])


def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    # Compute affine transform matrices
    warp_mat1 = cv2.getAffineTransform(np.float32(t1), np.float32(t))
    warp_mat2 = cv2.getAffineTransform(np.float32(t2), np.float32(t))

    # Warp triangles
    warped_img1 = cv2.warpAffine(img1, warp_mat1, (img.shape[1], img.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warped_img2 = cv2.warpAffine(img2, warp_mat2, (img.shape[1], img.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Mask for the triangle
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t), 1.0, 16, 0)

    # Blend the triangles
    img += mask * ((1.0 - alpha) * warped_img1 + alpha * warped_img2)

def morph_images(img1, img2, points1, points2, tri, alpha):
    h, w, c = img1.shape  
    morphed_img = np.zeros((h, w, c), dtype=np.float32)

    points = (1 - alpha) * points1 + alpha * points2

    for tri_indices in tri.simplices:
        x1 = points1[tri_indices]
        x2 = points2[tri_indices]
        x = points[tri_indices]

        # For each channel separately
        for ch in range(c):
            morph_triangle(img1[:,:,ch], img2[:,:,ch], morphed_img[:,:,ch], x1, x2, x, alpha)

    return np.clip(morphed_img, 0, 255).astype(np.uint8)

# from the landmarks output by ADNet (B x N_points x 2), in the range of (-1, 1), convert to coordinates in the image  
def get_actual_coordinates(h, w, landmarks):
    x_pixel = ((landmarks[:, :, 0] + 1) / 2) * w
    y_pixel = ((landmarks[:, :, 1] + 1) / 2) * h
    return torch.stack((x_pixel, y_pixel), dim=2).squeeze(0).cpu().numpy()

# function to initalize ADNet
def initialize_net(model_path, device=torch.device("cuda")):
    config = Alignment()

    net = stackedHGNetV1.StackedHGNetV1(classes_num=config.classes_num, \
                                        edge_info=config.edge_info, \
                                        nstack=config.nstack, \
                                        add_coord=config.add_coord, \
                                        pool_type=config.pool_type, \
                                        use_multiview=config.use_multiview)

    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint["net"])

    # send to gpu, set to evaluation mode
    net = net.float().to(device)
    net.eval()
    return net

# pass the image throguh ADNet to get landmarks in image coordinates
def get_landmarks_ADNet(img, net, device=torch.device("cuda")):
    old_h, old_w = img.shape[:2]
    # preprocess image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype('float32') / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # get landmarks
    with torch.no_grad():
        _, _, landmarks = net(img)
        landmarks = get_actual_coordinates(old_h, old_w, landmarks)

    return landmarks