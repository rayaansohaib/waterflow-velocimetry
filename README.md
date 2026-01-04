Water Surface Velocity Estimation Pipeline

This pipeline estimates water surface velocity (m/s) from a single monocular video
by tracking surface tracers (foam, bubbles, debris) and converting pixel motion
into real-world velocity.

The workflow consists of:
1. Calibration (one-time per camera/view)
2. Velocity estimation run (repeatable)

------------------------------------------------------------
1. CALIBRATION (REQUIRED ONCE PER CAMERA SETUP)
------------------------------------------------------------

Calibration defines:
- The region of interest (ROI) containing flowing water
- A homography to warp the view into a stable top-down plane
- A scale factor to convert pixels into meters

Calibration usage example:

python waterflow_pipeline.py calibrate \
  --video path/to/video.mp4 \
  --roi out/roi.json \
  --homography out/H.npy \
  --scale out/scale.json

This command launches interactive calibration steps.

------------------------------------------------------------
STEP 1 — ROI SELECTION
------------------------------------------------------------

You will be shown a video frame.

What to do:
- Click points around the water surface you want to analyze
- Exclude river banks, shadows, static objects, and reflections
- After constructing your polygon press enter


How to know it is correct:
- The ROI tightly covers only moving water
- No land or static features are included


Saved file:
out/roi.json

------------------------------------------------------------
STEP 2 — HOMOGRAPHY SELECTION (WARP SETUP)
------------------------------------------------------------

This step establishes the mapping used to warp the original camera view into a
stabilized/top-down plane where motion is measured consistently.

What to do:
- Select the required reference points (typically 4 points)
- Choose points that lie on the same physical plane as the water surface
- Prefer corners/markers/visible features that are stable and easy to click

How to know it is correct:
- After clicking enter you will be shown a warped roi view
- The warped view should not be heavily skewed or “bent”
- The ROI region looks rectangular/stable in the warped plane (not stretched oddly)


Saved file:
out/H.npy

------------------------------------------------------------
STEP 3 — SCALE SELECTION (METERS PER PIXEL)
------------------------------------------------------------

After the warp is established, you define the real-world scale.

What to do:
- In the warped view, click two points with a known real-world distance between them
  (e.g., 1 m, 2 m, spacing between markers, known width)
- press enter and then press d
- Enter the real distance in the command line when prompted

How to know it is correct:
- The chosen points are clearly identifiable and on the water plane
- The resulting speeds are realistic


Saved file:
out/scale.json

Calibration is now complete.

------------------------------------------------------------
2. VELOCITY ESTIMATION RUN
------------------------------------------------------------

Once calibration files exist, velocities can be computed from video alone.
An optional IMU input can be provided to assist with stabilization or
orientation correction, if available.

Run command (video only):

python waterflow_pipeline.py run \
  --video path/to/video.mp4 \
  --homography out/H.npy \
  --roi out/roi.json \
  --scale out/scale.json \
  --out_dir out/run1

Run command (video + optional IMU):

python waterflow_pipeline.py run \
  --video path/to/video.mp4 \
  --homography out/H.npy \
  --roi out/roi.json \
  --scale out/scale.json \
  --imu path/to/imu_data.csv \
  --out_dir out/run1

The IMU input is optional. If not provided, the pipeline operates purely
on vision-based motion estimation.

------------------------------------------------------------
3. OUTPUTS
------------------------------------------------------------

All outputs are written to the directory specified by --out_dir.

Typical outputs include:
- Velocity CSV file
  - Per-frame or per-track velocities in m/s
- Overlay video
  - Tracked features and velocity vectors drawn on the warped view
- Optional debug plots or intermediate images if enabled

If IMU data is provided:
- Stabilization or orientation-corrected motion estimates may be used
- IMU-aligned diagnostic plots or logs may also be generated

Example output directory:
out/run1/

------------------------------------------------------------
4. NOTES
------------------------------------------------------------

- Calibration only needs to be repeated if the camera position or view changes
- Poor tracer visibility, incorrect ROI, or bad homography points will degrade results
- IMU input is optional and intended to improve robustness
- Report detailing methedology and design is available in docs
  
