import cv2 as cv
import numpy as np

print("Identification of objects in a video\nby 180616T-P.M.P.H. Somarathne")

def get_indexed_image(im):
  """
    Thresholding, closing, and connected component anysis lumped
  """
  th, img = cv.threshold(im, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
  kernel = np.ones((3,3), dtype=np.uint8)
  closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
  retval, labels, stats, centroids = cv.connectedComponentsWithStats(closing)
  return retval, labels, stats, centroids

def is_new(a, b, delta, i):
  """
    Vector Dissimilarity with an Array of Vectors
    Checks if vector b is similar to a one or more vectors in a outside the tolerances specified in delta.
    vector i specified which elements in b to compare with those in a
  """
  if (np.absolute(a[:,i] - b[i]) > delta).all(): return True
  return False

def prev_index(a, b, delta, i):
  """
    Returns Previous Index
    Returns the index of the appearance of the object in the previous frame
  """
  index = np.where(np.absolute(a[:,i] - b[i]) <= delta)
  return index[0]

# Detect contours of template.png to be used as reference
print("Reading template")
template_im = cv.imread(r'template.png', cv.IMREAD_GRAYSCALE)
th_t, img_t = cv.threshold(template_im, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
kernel = np.ones((3,3), dtype=np.uint8)
closing_t = cv.morphologyEx(img_t, cv.MORPH_CLOSE, kernel)
contours_t, hierarchy_t = cv.findContours(closing_t, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Read the video
print("Reading video")
col_frames = []
frames = []
cap = cv.VideoCapture('conveyor_with_rotation.mp4') # give the correct path here
while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    col_frames.append(frame)
    frames.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
cap.release()
cv.destroyAllWindows()
frames = np.array(frames)
print("Frames shape:", frames.shape)

# Generate object flow
print("Generating object flow")
object_flow = []
matching_threshold = 4.5e-3
for grey in frames:
  retval, labels, stats, centroids = get_indexed_image(grey)
  contours, hierarchy = cv.findContours(((labels >= 1)*255).astype('uint8'), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  frame_objects = []
  for i in range(len(contours)):
    if cv.matchShapes(contours_t[0], contours[i], cv.CONTOURS_MATCH_I1, 0.0) > matching_threshold: continue
    ca = int(cv.contourArea(contours[i]))
    if ca<59500: continue
    M = cv.moments(contours[i])
    cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    frame_objects.append([cx, cy, ca, i+1])
  frame_objects = np.array(frame_objects)
  object_flow.append(frame_objects)

# Track the nut and assign identification number
print("Tracking the nuts")
delta_x = 15
prev_frame = object_flow[0]
obj_count = object_flow[0].shape[0]
for frame in object_flow[1:]:
  for obj in frame:
    if is_new(prev_frame, obj, 15, 0):
      obj_count+=1
      obj[3] = obj_count
    else:
      obj[3] = prev_frame[prev_index(prev_frame, obj, delta_x, 0)][0, 3]
  prev_frame = frame
print("Detected", obj_count, "nuts in the video")

# Add identification number into original image as text
print("Adding text")
for i in range(len(frames)):
  frame = col_frames[i]
  for obj in object_flow[i]:
    frame = cv.putText(frame, str(int(obj[3])), (int(obj[0]), int(obj[1])), cv.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
    frame = cv.putText(frame, '180616T', (20, 1060), cv.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)

# Encoding the frames
print("Encoding")
file_name = '180616t_en2550_a05.mp4'
fourcc = cv.VideoWriter_fourcc(*'MP4V')
out = cv.VideoWriter(file_name, fourcc, 30.0, (1920,  1080), True)
for frame in col_frames:
    out.write(frame)

# Release everything if job is finished
out.release()
print("Identification complete. You can view", file_name)
