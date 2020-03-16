#!/usr/bin/env python3

from os import path, makedirs, listdir
import os
os.environ["CDF_LIB"] = "/home/nmdung/my_libs/cdf37_0-dist-all/cdf37_0-dist/lib"
from shutil import move
from spacepy import pycdf
import numpy as np
import h5py
from subprocess import call
from tempfile import TemporaryDirectory
from tqdm import tqdm
from easydict import EasyDict as edict
from scipy import ndimage
import cv2

from metadata import load_h36m_metadata


metadata = load_h36m_metadata()
print('load metadata done...')
# Subjects to include when preprocessing
included_subjects = {
    'S1': 1,
    'S5': 5,
    'S6': 6,
    'S7': 7,
    'S8': 8,
    'S9': 9,
    'S11': 11,
}

# Sequences with known issues
blacklist = {
    ('S11', '2', '2', '54138969'),  # Video file is corrupted
}

# Define the list of 17 joints
h36m_keypoints = {
    0: 'Hip',
    1: 'RHip',
    2: 'RKnee',
    3: 'RFoot',
    6: 'LHip',
    7: 'LKnee',
    8: 'LFoot',
    12: 'Spine',
    13: 'Neck',
    14: 'Nose',
    15: 'Head',
    17: 'LShoulder',
    18: 'LElbow',
    19: 'LWrist',
    25: 'RShoulder',
    26: 'RElbow',
    27: 'RWrist',
}


# Rather than include every frame from every video, we can instead wait for the pose to change
# significantly before storing a new example.
def select_frame_indices_to_include(poses_3d):
    # To process every single frame, uncomment the following line:
    # return np.arange(0, len(poses_3d))

    # Down sample the frame rate from 50Hz to 10Hz, choose every 5th frames

    frame_indices = np.arange(1, len(poses_3d), 5)
    return frame_indices


def select_bboxes(bboxes_path, frame_indices):
    """
    Calculate the bounding box of human in the selected frames
    :param bboxes_path:
    :param frame_indices:
    :return:
    """
    print('process bounding boxes at {}'.format(bboxes_path))
    bboxes = {}
    f = h5py.File(bboxes_path, 'r')
    mask = f['Masks']
    for frame_idx in frame_indices:
        st = mask[frame_idx][0]
        obj_arr = np.transpose(np.array(f[st]))
        box = ndimage.find_objects(obj_arr)
        xmax, xmin = box[0][1].stop, box[0][1].start
        ymax, ymin = box[0][0].stop, box[0][0].start
        bboxes['img_%06d' % frame_idx] = [xmin, ymin, xmax, ymax]  # VOC format
    return bboxes


def select_3d_mono_view(mono_path, frame_indices, type='poses_3d_mono'):
    """
    We select 3D poses and angles annotations from 1 view based on the selected frames
    :param mono_path: path of annotation for each view
    :param frames: the selected frames
    :param type: poses or angles
    :return:
    """
    cdf = pycdf.CDF(mono_path)
    ret_mono_3d = {}
    mono_3d = np.array(cdf['Pose'])
    if type.startswith('pose'):
        mono_3d = mono_3d.reshape(mono_3d.shape[1], 32, 3)
        selected_keypoint_ids = list(h36m_keypoints.keys())
        mono_3d = mono_3d[:, selected_keypoint_ids, :]
    elif type.startswith('angle'):
        mono_3d = mono_3d.reshape(mono_3d.shape[1], 26, 3)

    for frame_idx in frame_indices:
        ret_mono_3d['img_%06d' % frame_idx] = mono_3d[frame_idx]
    return ret_mono_3d


def select_2d_mono_view(mono_path, frame_indices):
    """
    We select 2D poses annotations from 1 view based on the selected frames
    :param mono_path: path of annotation for each view
    :param frames: the selected frames
    :param type: poses or angles
    :return:
    """
    cdf = pycdf.CDF(mono_path)
    ret_mono_2d = {}
    mono_2d = np.array(cdf['Pose'])
    mono_2d = mono_2d.reshape(mono_2d.shape[1], 32, 2)  # numframe, 32, 2
    selected_keypoint_ids = list(h36m_keypoints.keys())
    mono_2d = mono_2d[:, selected_keypoint_ids, :]
    for frame_idx in frame_indices:
        ret_mono_2d['img_%06d' % frame_idx] = mono_2d[frame_idx]
    return ret_mono_2d


def process_view(out_dir, subj_dir, subject, action, subaction, camera, frame_indices):
    base_filename = metadata.get_base_filename(subject, action, subaction, camera)

    video_path = path.join(subj_dir, 'Videos', base_filename + '.mp4')
    frames_dir = path.join(out_dir, 'imageSequence', camera)
    makedirs(frames_dir, exist_ok=True)

    # Check to see whether the frame images have already been extracted previously
    existing_files = {f for f in listdir(frames_dir)}
    frames_already_extracted = False
    for frame_idx in frame_indices:
        filename = 'img_%06d.jpg' % frame_idx
        if filename in existing_files:
            frames_already_extracted = True
            break
    # Check to extract images only 1 time

    if not frames_already_extracted:
        print('process video: {}'.format(video_path))
        video_cap = cv2.VideoCapture(video_path)
        f_idx = -1
        while True:
            ret, image = video_cap.read()
            if ret:
                f_idx += 1
                if f_idx in frame_indices:
                    filename = 'img_%06d.jpg' % f_idx
                    cv2.imwrite(path.join(frames_dir, filename), image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        video_cap.release()
    else:
        print('The images are already extracted')

    bboxes_path = path.join(subj_dir, 'Bboxes', '{}.mat'.format(base_filename))
    poses_3d_mono_path = path.join(subj_dir, 'Poses_D3_Positions_mono', '{}.cdf'.format(base_filename))
    poses_3d_mono_uni_path = path.join(subj_dir, 'Poses_D3_Positions_mono_universal', '{}.cdf'.format(base_filename))
    angles_3d_mono_path = path.join(subj_dir, 'Angles_D3_mono', '{}.cdf'.format(base_filename))
    poses_2d_mono_path = path.join(subj_dir, 'Poses_D2_Positions', '{}.cdf'.format(base_filename))

    bboxes = select_bboxes(bboxes_path, frame_indices)
    poses_3d_mono = select_3d_mono_view(poses_3d_mono_path, frame_indices, type='poses_3d_mono')
    poses_3d_mono_uni = select_3d_mono_view(poses_3d_mono_uni_path, frame_indices, type='poses_3d_mono_uni')
    angles_3d_mono = select_3d_mono_view(angles_3d_mono_path, frame_indices, type='angles_3d_mono')
    poses_2d_mono = select_2d_mono_view(poses_2d_mono_path, frame_indices)

    return bboxes, poses_3d_mono, poses_3d_mono_uni, angles_3d_mono, poses_2d_mono


def process_subaction(datasets_dir, subject, action, subaction):
    out_dir = path.join(datasets_dir, 'human36M', 'protocol_1', subject, metadata.action_names[action] + '_' + subaction)
    makedirs(out_dir, exist_ok=True)
    annos_dir = path.join(out_dir, 'annos')
    makedirs(annos_dir, exist_ok=True)

    subj_dir = path.join(datasets_dir, 'human36M', 'extracted', subject)
    base_actname = metadata.get_base_actname(subject, action, subaction)
    # print('base_actname: {}'.format(base_actname))
    # Load joint position annotations
    poses_3d_cdf = pycdf.CDF(path.join(subj_dir, 'Poses_D3_Positions', base_actname + '.cdf'))
    poses_3d = np.array(poses_3d_cdf['Pose'])
    poses_3d = poses_3d.reshape(poses_3d.shape[1], 32, 3)

    angles_3d_cdf = pycdf.CDF(path.join(subj_dir, 'Angles_D3', base_actname + '.cdf'))
    angles_3d = np.array(angles_3d_cdf['Pose'])
    angles_3d = angles_3d.reshape(angles_3d.shape[1], 26, 3)

    frame_indices = select_frame_indices_to_include(poses_3d)
    # frame_indices += 1

    bboxes, poses_3d_mono, poses_3d_mono_uni, angles_3d_mono, poses_2d_mono = {}, {}, {}, {}, {}
    #extract images
    for camera in tqdm(metadata.camera_ids, ascii=True, leave=False):
        if (subject, action, subaction, camera) in blacklist:
            continue
        # try:
        sub_bboxes, sub_poses_3d_mono, sub_poses_3d_mono_uni, sub_angles_3d_mono, sub_poses_2d_mono = process_view(out_dir, subj_dir,
                                                                                                subject, action,
                                                                                                subaction, camera,
                                                                                                frame_indices)
        # except:
        #     print('Error processing sequence, skipping: ', repr((subject, action, subaction, camera)))
        #     continue
        bboxes['{}'.format(camera)] = sub_bboxes
        poses_3d_mono['{}'.format(camera)] = sub_poses_3d_mono
        poses_3d_mono_uni['{}'.format(camera)] = sub_poses_3d_mono_uni
        angles_3d_mono['{}'.format(camera)] = sub_angles_3d_mono
        poses_2d_mono['{}'.format(camera)] = sub_poses_2d_mono

    #Save annotations
    annos = {}
    for frame_idx in frame_indices:
        anno_bboxes_dict, anno_poses_3d_mono_dict, anno_poses_3d_mono_uni_dict, anno_angles_3d_mono_dict, anno_poses_2d_mono = {}, {}, {}, {}, {}
        for camera, sub_bboxes in bboxes.items():
            anno_bboxes_dict['{}'.format(camera)] = sub_bboxes['img_%06d' % frame_idx]
            anno_poses_3d_mono_dict['{}'.format(camera)] = poses_3d_mono['{}'.format(camera)]['img_%06d' % frame_idx]
            anno_poses_3d_mono_uni_dict['{}'.format(camera)] = poses_3d_mono_uni['{}'.format(camera)]['img_%06d' % frame_idx]
            anno_angles_3d_mono_dict['{}'.format(camera)] = angles_3d_mono['{}'.format(camera)]['img_%06d' % frame_idx]
            anno_poses_2d_mono['{}'.format(camera)] = poses_2d_mono['{}'.format(camera)]['img_%06d' % frame_idx]
        annos['img_%06d' % frame_idx] = {
            'bboxes': anno_bboxes_dict,
            'poses_3d': poses_3d[frame_idx],
            'poses_3d_mono': anno_poses_3d_mono_dict,
            'poses_3d_mono_uni': anno_poses_3d_mono_uni_dict,
            'poses_2d_mono': anno_poses_2d_mono,
            # 'angles_3d': angles_3d[frame_idx],
            # 'angles_3d_mono': anno_angles_3d_mono_dict,
        }
    annos_file = path.join(annos_dir, 'selected_annotations.npy')
    np.save(annos_file, annos)

def process_all(datasets_dir):
    sequence_mappings = metadata.sequence_mappings

    subactions = []

    for subject in included_subjects.keys():
        subactions += [
            (subject, action, subaction)
            for action, subaction in sequence_mappings[subject].keys()
            if int(action) > 1  # Exclude '_ALL'
        ]
    for subject, action, subaction in tqdm(subactions, ascii=True, leave=False):
        process_subaction(datasets_dir, subject, action, subaction)


if __name__ == '__main__':
    datasets_dir = '/media/nmdung/SSD_4TB_Disk_1/kpts_works/hpe_3D/datasets'
    process_all(datasets_dir)
