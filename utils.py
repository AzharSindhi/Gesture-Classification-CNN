import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import scipy.spatial.distance as dist

ec = [
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 0, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 0, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 0, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 0, 255),
]

ec  = [(255, 0, 255), (122, 122, 255), (255, 0, 0), (0, 255, 0)]

colors_hp = [(255, 0, 255), (122, 122, 255), (255, 0, 0), (0, 255, 0),  (0, 0, 255), (255, 122, 122), (215, 255, 0)]

np.random.shuffle(colors_hp)
# array([
#        ['nose_x', 'nose_y'] 0,
#        ['left_wrist_x', 'left_wrist_y'] 1,
#        ['right_wrist_x', 'right_wrist_y'] 2,
#        ['left_shoulder_x', 'left_shoulder_y'] 3,
#        ['right_shoulder_x', 'right_shoulder_y'] 4,
#        ['left_elbow_x', 'left_elbow_y'] 5,
#        ['right_elbow_x', 'right_elbow_y'] 6
# ], dtype='<U16')

edges = [
    # [0,1], [0, 2],
    [1, 5],
    [2, 6],
    [3, 5],
    [4, 6],
]

def hex2rgb(h):  # rgb order (PIL)
    color = tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))[::-1]
    return color # cv

def rescale_keypoints(keypoints, new_shape):
    pass


def calculate_distances(keypoints):
    pass


def draw_keypoints(img, points):

    points = points.astype(int)
    # for i, point in points:
    #     print('drawing point x, y', point, img.shape)
    #     cv2.circle(img, (point), 3, (0, 0, 255), -1)

    num_joints = points.shape[0]
    for j in range(num_joints):
        cv2.circle(img, (points[j, 0], points[j, 1]), 10, colors_hp[j], -1)
    for j, e in enumerate(edges):
        if points[e].min() > 0:
            cv2.line(
                img,
                (points[e[0], 0], points[e[0], 1]),
                (points[e[1], 0], points[e[1], 1]),
                ec[j],
                2,
                lineType=cv2.LINE_AA,
            )
    return img


def draw_bbox(img, bbox, text):
    # bbox[2] += bbox[0]
    # bbox[3] += bbox[1]
    img = cv2.rectangle(
        img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness=2
    )
    x = bbox[0]
    y = bbox[1]
    x_offset = 0  # int((bbox[2] - bbox[0])/2)
    y_offset = -10  # int((bbox[3] - bbox[1])/2)
    img = cv2.putText(
        img,
        text,
        (x + x_offset, y + y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (102, 255, 102),
        2,
    )
    return img


def str2list(inp):
    inp = inp.replace("[ ", "[")
    inp = inp.replace(" ]", "]")
    inp = inp.replace("[", "")
    inp = inp.replace("]", "")
    inp = inp.replace("  ", " ")
    out = inp.split()
    return out


def get_cropped_image(img, bbox):
    x1, y1, x2, y2 = bbox
    cropped_image = img[y1:y2, x1:x2]
    return cropped_image


def rescale_keypoints(points, original_shape, target_shape, order='wh'):
    if order == 'wh':
        orig_w, orig_h = original_shape[:2]
        new_w, new_h = target_shape[:2]
    if order == 'hw':
        orig_h, orig_w = original_shape[:2]
        new_h, new_w = target_shape[:2]
    
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    repeat = points.shape[0]
    scale = np.tile([scale_x, scale_y], (repeat, 1))
    points_scaled = points * scale
    points_scaled = points_scaled.astype(int)
    points_scaled[points_scaled < 0] = 0
    return points_scaled


def calculate_angles(points1, points2):
    costheta = 1 - dist.cdist(points1, points2, "cosine")
    return np.arccos(costheta)


def calculate_2Ddistances(points):
    """
    calculate pairwise distances
    """
    distances = euclidean_distances(points)
    return distances


def keypoints_to_image(points, input_shape, features_shape):
    points = points.astype(int)
    points = points[points >= 0]
    points = points[points < input_shape[0] - 1]
    img = np.zeros(input_shape)
    img[points] = 1
    img = np.resize(img, features_shape)
    img = img[np.newaxis, :]
    return img


def calculate_1D_distances(points):
    nose_points = [points[0]]
    left_elbow_point = [points[5]]
    right_elbow_point = [points[6]]
    distances = euclidean_distances(points, nose_points).squeeze()
    lelbow_angles = calculate_angles(points, left_elbow_point).squeeze()
    relbow_angles = calculate_angles(points, right_elbow_point).squeeze()
    keypoint_features = np.concatenate(
        (distances, lelbow_angles, relbow_angles))
    return keypoint_features


def calculate_1D_distances_nose(points, target_len=21):
    nose_points = points[0]
    distances = []
    angles = []
    for i in range(1, len(points) - 1, 2):
        ldist = dist.euclidean(nose_points, points[i])
        rdist = dist.euclidean(nose_points, points[i + 1])
        rnose_point = np.array(nose_points).reshape(1, 2)
        lpoint = np.array(points[i]).reshape(1, 2)
        rpoint = np.array(points[i + 1].reshape(1, 2))
        langle = calculate_angles(rnose_point, lpoint)[0][0]
        rangle = calculate_angles(rnose_point, rpoint)[0][0]
        min_angle = min(langle, rangle)
        dst = min(ldist, rdist)
        angles.append(min_angle)
        distances.append(dst)

    distances.extend(angles)
    # distances = normalize0_1(distances)
    repeat = int(target_len / len(distances)) + 1
    distances = np.tile(distances, repeat)
    distances = distances[:target_len]
    return distances


def prepare_2D_features(points, target_shape):
    """
    convert 1D features to 2D for CNN
    """
    pairwise_distances = calculate_2Ddistances(points)
    pairwise_angles = calculate_angles(points, points)
    features = np.concatenate((pairwise_distances, pairwise_angles), axis=0)
    features = cv2.resize(features, target_shape)
    features = np.expand_dims(features, axis=2).astype(int)
    return features


def normalize0_1(distances):
    minn = np.min(distances)
    maxx = np.max(distances)
    normalized_dist = (distances - minn) / (maxx - minn)
    return normalized_dist


def resize_image(image, bbox, feature_points, input_shape):
    input_shape = input_shape[:2]
    if isinstance(bbox, str):
        bbox = str2list(bbox)
        bbox = np.array(bbox).astype(float)

    bbox = np.round(bbox).astype(int)
    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]

    orig_h, orig_w = cropped_image.shape[:2]
    feature_points = np.array(feature_points).reshape(-1, 2)  # keypoints
    repeat = feature_points.shape[0]
    offset_term = np.tile([x1, y1], (repeat, 1))
    feature_points_cropped = feature_points - offset_term
    # rescale the points according to input shape
    new_w, new_h = input_shape
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    scale = np.tile([scale_x, scale_y], (repeat, 1))
    feature_points_cropped_scaled = feature_points_cropped * scale
    feature_points_cropped_scaled[feature_points_cropped_scaled < 0] = 0
    feature_points_cropped_scaled[feature_points_cropped_scaled > input_shape[0]] = (
        input_shape[0] - 1
    )

    cropped_image_resized = cv2.resize(cropped_image, (new_w, new_h))
    return cropped_image_resized, feature_points_cropped_scaled
