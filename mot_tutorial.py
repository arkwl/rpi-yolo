import sys

motdata = join(MOT_PATH,'train/MOT17-09/img1/')
sys.path.append(motdata)

import matplotlib.pylab as plt
import cv2

list_motdata = os.listdir(motdata)
list_motdata.sort()

img_ex_path = motdata + list_motdata[0]
img_ex_origin = cv2.imread(img_ex_path)
img_ex = cv2.cvtColor(img_ex_origin, cv2.COLOR_BGR2RGB)

plt.imshow(img_ex)
plt.axis('off')
plt.show()

