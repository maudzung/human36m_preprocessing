HUMAN3.6M DATASET AND CODE ORGANIZATION

The available data is organized by subject and by motion and activity
scenarios. There are test subjects and training subjects. The training
data consists of videos, segments, features, Time-of-Flight data, meshes
and poses. For the testing part we provide only video, segment and feature
data. Evaluation is only available through our server.

N.B. Due to certain privacy concerns, we have decided to withhold the
images of one of our testing subjects, S10. However, we make all the
other data associated to this subject available, including silhouettes,
bounding boxes as well as corresponding image descriptors. In the future,
as other features are developed by external researchers or by us, we will
strive to compute those and make them available for download for S10.
Error evaluation on the test set is possible both with and without S10.
For additional info on S10, please have a look at the Tech Report on the
home page.

For the training data, there are 15 scenarios. For each one of them there
are several data streams available: videos, segments, features, range data
(TOF) and poses.

The following section describes into some detail each category of data.

SUBMISSION AND EVALUATION 

After downloading the code zip file(Version 1.1) follow the README instructions.

Note: a submission file that contains NaN or Inf values, in any of the predictions
associated with an action, will not have an entry in the ranking table for the
corresponding action. This way, a user can choose to evaluate only on a subset of
action classes.

DATA CATEGORIES

VIDEOS - Are provided in a high quality compressed MPEG4 format, 50fps.

SEGMENTS - These are the segments obtained using background subtraction. They
are provided either in a compact Matlab format or MPEG4 format (the latter is
meant for users who do not use Matlab(r)). 

N.B. The video encoder introduces some artifacts so we recommend to use the
.mat files whenever possible.

FEATURES - we provide several image features pre-computed (also used in our
experiments). More may be added, but these are for now pyramid HoG features
extracted on either bounding boxes or our background subtraction segments. Note
that these features were extracted on non-compressed video data, so they may
differ somewhat from the features extracted on the MPEG4 videos. The features
are provided in CDF format (Common Data Format), a format supported by Matlab
and many APIs for different other languages and architectures.

TOF - Time-of-Flight data obtained using a Mesa Imaging SR4000 camera are also
provided in CDF format. The archives contain both depth and intensity data.
Range data is synchronized with the images and poses.

SCANNER - The 3D meshes of the subjects, obtained with a 3D laser scan, are
provided in Wavefront's obj format, a standard text based format which is
supported on many platforms. For Matlab we provide an API that allows loading
this type of file. We also provide the original point clouds in txt format
without gap filling or mesh smoothing, for those interested.

POSES - We provide the poses in different parametrizations (all in CDF file
format). The Vicon exported data is provided as RawAngles on our website. This
should be enough for Matlab users who have access to Matlab's 3D Animation
toolbox (the toolbox is needed because we provide code that uses it to help
compute all the other pose parametrizations). For those who do not have access
to the 3D Animation toolbox, we also provide precomputed files. The
parametrizations we provide are 3D positions in the original coordinate space
(D3_Positions) and transformed for monocular prediction using the camera
parameters (D3_Positions_mono). We also provide 3D Angles for monocular
prediction (D3_Angles_mono) and projections of the skeleton onto the image
plane (D2_Positions). Lastly we provide 3D positions using the same limb
lengths for all subjects (D3_Positions_mono_universal) as a 3D position
parametrization that is invariant to subject size. The skeleton information is
provided in the metadata.xml file that is delivered with our code.

MIXED REALITY
We provide compressed mixed reality test videos where graphics characters
were animated with human motion capture and rendered on real background
using geometrically correct models, camera motion estimation, etc. 

VISUALIZATION AND LARGE SCALE PREDICTION SOFTWARE
We provide Matlab code to visualize and manipulate the data, including
skeletons, as well as large scale continuous prediction models. We provide
nearest neighbor, as well as both standard and structured regression models,
made scalable by means of Fourier approximations to non-linear kernels.
Computing the results on the withheld test data uses our evaluation server
with weekly timers to prevent the over usage of the test set. See 'Your
Results' section on our website.