import cv2 as cv
import numpy as np

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

template_im = cv.imread(r'template.png', cv.IMREAD_GRAYSCALE)
retval, labels, stats, centroids = get_indexed_image(template_im)
contours_t, hierarchy_t = cv.findContours(((labels >= 1)*255).astype('uint8'), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

frames = []
cap = cv.VideoCapture('conveyor_with_rotation.mp4') # give the correct path here
while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret: break
    frames.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
cap.release()
cv.destroyAllWindows()
frames = np.array(frames)

print("Frames shape:", frames.shape)

object_flow = []
matching_threshold = 0.00018
for grey in frames:
  retval, labels, stats, centroids = get_indexed_image(grey)
  contours, hierarchy = cv.findContours(((labels >= 1)*255).astype('uint8'), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  frame_objects = []
  for i in range(len(contours)):
    if cv.matchShapes(contours_t[0], contours[i], cv.CONTOURS_MATCH_I1, 0.0) > matching_threshold: continue
    ca = cv.contourArea(contours[i])
    M = cv.moments(contours[i])
    cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
    frame_objects.append([cx, cy, ca, i])
  frame_objects = np.array(frame_objects)
  object_flow.append(frame_objects)

delta_x = 200
prev_frame = object_flow[0]
obj_count = prev_frame.shape[0]
for frame in object_flow[1:]:
  for obj in frame:
    if is_new(prev_frame, obj, delta_x, 0):
      obj_count+=1
  prev_frame = frame

print("Found", obj_count, "nuts in the video")
input()
  
