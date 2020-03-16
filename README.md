This instruction will help you to pre-process [the Human3.6M dataset](http://vision.imar.ro/human3.6m/description.php).

The source code is referred to [h36m-fetch](https://github.com/anibali/h36m-fetch) repository. 
However, I used OpenCV to extract images from videos instead of ffmpeg. 
This leads to a better quality of extracted images (3 times).

**Folder structure**
```
    ${ROOT}
    ├──datasets/
    ├──human36m_preprocessing/
        ├──download_all.py
        ├──extract_all.py
        ├──metadata.py
        ├──metadata.xml
        ├──process_all.py
        ├──protocol_1.py
        ├──protocol_1_selected_annos.py
        ├──README.md
        ├──README_from_authors.md
    ├──src/
```

The full steps are below:

### 1. Download the dataset from Human3.6m webpage
```python
    python3 download_all.py
```

### 2. Extract the downloaded files
```python
    python3 extract_all.py
```

### 3. Extract images from original videos and get the _full_ ground-truth
```python
    python3 protocol_1.py
```
if you want to extract only the annotations of selected joints, you can run the command:
```python
    python3 protocol_1_selected_annos.py
```

### 4. Extract images from original videos and get the selected ground-truth
```python
    python3 protocol_1_selected_annos.py
```
The list of selected keypoints in Human3.6M dataset
```python
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
```

**Protocol 1**:
- Training subjects: S1, S5, S6, S7, S8
- Testing subjects: S9, S11
- Down-sample from 50Hz to 10Hz for every subjects

**Protocol 2**: 
- Training subjects: S1, S5, S6, S7, S8, S9
- Testing subjects: S11
- Test on every 64th frames in the original videos

## Annotations structure
```python
annot_dict = {
    'S1': {
        'action-subaction': {
            'frame': {
                '3d_poses': 1,
                '3d_angles': 2,
                '3d_bboxes': 3,
            },
            'frame': {
                '3d_poses': 1,
                '3d_angles': 2,
                '3d_bboxes': {
                    'camera_1': ['xmin', 'ymin', 'xmax', 'ymax'],
                    'camera_2': ['xmin', 'ymin', 'xmax', 'ymax'],
                },
            },
        },
        'action-subaction_2': {
            'frame': {
                '3d_poses': 1,
                '3d_angles': 2,
                '3d_bboxes': 3,
            },
            'frame': {
                '3d_poses': 1,
                '3d_angles': 2,
                '3d_bboxes': 3,
            }
        }
    }
}
```