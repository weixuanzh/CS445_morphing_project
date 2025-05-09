import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import torch
import matplotlib
import sys
sys.path.append('./')
# from external.ADNet.lib.backbone import stackedHGNetV1
from external.ADNet.lib.backbone import stackedHGNetV1
from external.ADNet.conf.alignment import Alignment
import mediapipe as mp

# matplotlib.use('TkAgg')

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

def run_symmetry_accuracy_test(): #Used to measure the symmetry or bias in the interpolation done by the morph images function
    #If both images are the same for A->B and B->A at 0.5, that means there is no inconsistencies and the warping is accurate

    folder_path = './image_set'
    image_names = [name for name in os.listdir(folder_path) if name.lower().endswith('.jpg')]
    loaded_images = [ cv2.resize((cv2.cvtColor(cv2.imread(folder_path + '/' + name),cv2.COLOR_BGR2RGB)), (128,128)) for name in image_names]
    #loaded_images = [ cv2.imread(folder_path + '/' + name) for name in image_names]

    face_mesh_tools = mp.solutions.face_mesh
    face_mesh = face_mesh_tools.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5)
    important_landmarks = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,361,288, 397, 365, 379, 378, 400, 152, 148, 176, 136, 172,58, 132, 234, 127, 162, 21, 54, 103, 67, 109, 33, 133, 159, 145, 160, 144, 153, 154, 155,362, 263, 386, 374, 387, 373, 380, 381, 382,1, 2, 98, 327, 94, 331, 168, 197, 195, 5, 4,61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 78, 95, 88, 178]
    facial_points = []
    no_face = []
    count_for_removal = 0
    
    for l_image in loaded_images:
        landmarks = face_mesh.process(l_image)
        height,width,_ = l_image.shape
    
        if landmarks.multi_face_landmarks: #if any face at all is detected, we select the first one
            facial_landmarks = landmarks.multi_face_landmarks[0] #they are spit out as a normalized value so we have to reintroduce them
            important_points_per_image = []
            for il in important_landmarks:
                x,y =  int(facial_landmarks.landmark[il].x * width), int(facial_landmarks.landmark[il].y * height)
                important_points_per_image.append((x,y))
            facial_points.append(important_points_per_image)
        else:
            no_face.append(count_for_removal)
        count_for_removal += 1

    for entry in sorted(no_face, reverse=True): #get rid of images that contain no detectable face
        loaded_images.pop(entry)
        no_face.remove(entry)

    error = 0
    
    for j in range(len(loaded_images)):
        first_image = loaded_images[j]
        first_points1 = facial_points[j]
        boundary_points = add_boundary_points(first_image.shape)
        first_points = np.vstack([first_points1,boundary_points])
        
        for i in range(len(loaded_images)):
            if i == j:
                continue
            curr_image = loaded_images[i]
            curr_points = np.vstack([facial_points[i],boundary_points])
            avg_points = (first_points + curr_points) / 2
            triangular = Delaunay(avg_points)
            
            pt1 = morph_images(first_image, curr_image, first_points, curr_points, triangular, 0.5)
            pt2 = morph_images(curr_image, first_image, curr_points, first_points, triangular, 0.5)
            error += np.mean(np.abs(pt1.astype(np.float32) - pt2.astype(np.float32)))
            
    total_comparisons = len(loaded_images) * (len(loaded_images) - 1)
    return (100 - error / total_comparisons)

def top_closest_points(available_points,point,n=10):
    distance = np.sqrt(np.sum((available_points - point) ** 2, axis=1))
    toreturn = available_points[distance <= n]
    return toreturn

def get_canny_points(img1, interval=50): #DOESN'T WORK WITH THE IMAGE MORPHING AT ALL, REMOVED FROM THE WRAPPER
    img = img1.copy()
    if img.dtype != "uint8":
        img = (255 * img).clip(0, 255).astype("uint8")
    if img is None:
        raise ValueError("Image failed to load. Check the path or file.")

    edges = cv2.Canny(img,100,200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected_points = select_points(img, title="Select primary points")
    flattened = np.vstack([c.reshape(-1, 2) for c in contours])

    sampled_points = np.empty((0, 2))
    for pt in selected_points:
        returnStart = top_closest_points(flattened,pt)
        sampled_points = np.vstack((sampled_points, returnStart))

    more_sampled = flattened[::interval]

    final_points = np.vstack((selected_points,sampled_points,more_sampled))

    return np.array([tuple(point) for point in final_points])

def get_mp_points(img):
    face_mesh_tools = mp.solutions.face_mesh
    face_mesh = face_mesh_tools.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5)
    important_landmarks = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,361,288, 397, 365, 379, 378, 400, 152, 148, 176, 136, 172,58, 132, 234, 127, 162, 21, 54, 103, 67, 109, 33, 133, 159, 145, 160, 144, 153, 154, 155,362, 263, 386, 374, 387, 373, 380, 381, 382,1, 2, 98, 327, 94, 331, 168, 197, 195, 5, 4,61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 78, 95, 88, 178]
    facial_points = []
    landmarks = face_mesh.process(img)
    height,width,_ = img.shape

    if landmarks.multi_face_landmarks: #if any face at all is detected, we select the first one
        facial_landmarks = landmarks.multi_face_landmarks[0] #they are spit out as a normalized value so we have to reintroduce them
        for il in important_landmarks:
            x,y =  int(facial_landmarks.landmark[il].x * width), int(facial_landmarks.landmark[il].y * height)
            facial_points.append((x,y))
    else:
        return None
        
    return np.array(facial_points)

def Image_Morphing_Video(imgA,imgB,videoname='morph_video.mp4',point_selection='MANUAL'):
    possible_point_selections = ['MANUAL', 'MP', 'ADNET']
    
    if point_selection not in possible_point_selections:
        print(f"Invalid point_selection method. Choose from: {possible_point_selections}")
        return
        
    if len(imgA.shape) != len(imgB.shape):
        print("Both Images Aren't Using the Same Color Channels")
        return

    imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))
    
    points1 = []
    points2 = []
    tri = []
    boundary_points = add_boundary_points(imgA.shape)
    h,w,_ = imgA.shape
    
    if point_selection == 'MANUAL':
        points1 = np.vstack([select_points(imgA),boundary_points])
        points2 = np.vstack([select_points(imgB),boundary_points])
        
    elif point_selection == 'MP':
        mp_points_1 = get_mp_points(imgA)
        if mp_points_1 is None:
            print("imgA Has No Detectable Facial Structure, Please Use ADNET or MANUAL")
            return None
        mp_points_2 = get_mp_points(imgB)
        if mp_points_2 is None:
            print("imgB Has No Detectable Facial Structure, Please Use ADNET or MANUAL")
            return None
        points1 = np.vstack([mp_points_1,boundary_points])
        points2 = np.vstack([mp_points_2,boundary_points])
        
    elif point_selection == 'ADNET':
        #TODO?
        print("Not Sure How to Get the ADNET DATA In?")

    if points1.shape[0] != points2.shape[0]:
        diff = abs(points1.shape[0] - points2.shape[0])
        if points1.shape[0] < points2.shape[0]:
            extra = points2[-diff:]
            points1 = np.vstack([points1, extra])
        else:
            extra = points1[-diff:]
            points2 = np.vstack([points2, extra])

    avg_points = (points1 + points2) / 2
    tri = Delaunay(avg_points)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(videoname, fourcc, 15.0, (w, h), isColor=True)

    num_frames = 100  # Forward frames
    frames = []
    
    for i, alpha in enumerate(np.linspace(0, 1, num_frames)):
        print(f"Generating forward frame {i+1}/{num_frames}...")
        frame = morph_images(imgA, imgB, points1, points2, tri, alpha)
        frames.append(frame)
    
    for frame in frames:
        video_out.write(frame)
    
    # Write backward frames (skip last frame to avoid repeated frame)
    for frame in frames[-2::-1]:  # Start second to last frame, reverse
        video_out.write(frame)
    video_out.release()
    print(f"video saved as {videoname} ")

    cap = cv2.VideoCapture(videoname)

    if not cap.isOpened():
        print("Error opening video file")
    
    print("Playing video... Press Enter in the terminal to exit.")
    
    while True:
        ret, frame = cap.read()
    
        if not ret:
            # If video ended, restart from beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
    
        cv2.imshow('Morph Video Preview', frame)
    
        # Check for keypress every 30ms
        if cv2.waitKey(70) == 13:  # 13 is Enter key
            break
    
    cap.release()
    cv2.destroyAllWindows()

def Image_Morphing_Image(imgA,imgB,alpha=0.8,point_selection='MANUAL'):
    possible_point_selections = ['MANUAL', 'MP', 'ADNET']
    
    if point_selection not in possible_point_selections:
        print(f"Invalid point_selection method. Choose from: {possible_point_selections}")
        return
        
    if len(imgA.shape) != len(imgB.shape):
        print("Both Images Aren't Using the Same Color Channels")
        return

    imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))
    
    points1 = []
    points2 = []
    tri = []
    boundary_points = add_boundary_points(imgA.shape)
    
    if point_selection == 'MANUAL':
        points1 = np.vstack([select_points(imgA),boundary_points])
        points2 = np.vstack([select_points(imgB),boundary_points])
        
    elif point_selection == 'MP':
        mp_points_1 = get_mp_points(imgA)
        if mp_points_1 is None:
            print("imgA Has No Detectable Facial Structure, Please Use ADNET or MANUAL")
            return None
        mp_points_2 = get_mp_points(imgB)
        if mp_points_2 is None:
            print("imgB Has No Detectable Facial Structure, Please Use ADNET or MANUAL")
            return None
        points1 = np.vstack([mp_points_1,boundary_points])
        points2 = np.vstack([mp_points_2,boundary_points])
        
    elif point_selection == 'ADNET':
        #TODO?
        print("Not Sure How to Get the ADNET DATA In?")

    if points1.shape[0] != points2.shape[0]:
        diff = abs(points1.shape[0] - points2.shape[0])
        if points1.shape[0] < points2.shape[0]:
            extra = points2[-diff:]
            points1 = np.vstack([points1, extra])
        else:
            extra = points1[-diff:]
            points2 = np.vstack([points2, extra])
        
        
    avg_points = (points1 + points2) / 2
    tri = Delaunay(avg_points)

    return morph_images(imgA, imgB, points1, points2, tri, alpha)