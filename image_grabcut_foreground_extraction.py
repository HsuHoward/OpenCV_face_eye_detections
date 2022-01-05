import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
# steps of Grabcut
# 1. define a rectangle in an image
# 2. the area outside of the rectangle is automatically considered as background
# 3. use the background info to segregate the background and figure in the rectangle is
# 4. use Gaussian mixed model (GMM) to model the background and figure, and denote the undefined pixels as the possible background/figure
# 5. the adjacent pixels are considered as a graph structure and the links are the color similarity between the adjacent pixels
# 6. every pixel in the graph structure is linked to the node of figure/background
# 7. if the linked nodes (may linked to figure/background) belong to different end nodes, then cut their edge for segregate the image
GrabCut演算法的實現步驟：   
1 在圖片中定義(一個或者多個)包含物體的矩形。
2 矩形外的區域被自動認為是背景。
3 對於使用者定義的矩形區域，可用背景中的資料來區分它裡面的前景和背景區域。
4 用高斯混合模型(GMM)來對背景和前景建模，並將未定義的畫素標記為可能的前景或者背景。
5 影象中的每一個畫素都被看做通過虛擬邊與周圍畫素相連線，而每條邊都有一個屬於前景或者背景的概率，這是基於它與周邊畫素顏色上的相似性。
6 每一個畫素(即演算法中的節點)會與一個前景或背景節點連線。
7 在節點完成連線後(可能與背景或前景連線)，若節點之間的邊屬於不同終端(即一個節點屬於前景，另一個節點屬於背景)，則會切斷他們之間的邊，這就能將影象各部分分割出來。

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount[, mode]) → None
'''

img = cv2.imread('humen.jpeg')
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgModel = np.zeros((1, 65), np.float64)

rect = (100, 10, 50, 70)  # (dx, dy, weight, hight)

cv2.grabCut(img, mask, rect, bgdModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
# 呼叫grabcut得到rect[0,1,2,3],將0,2合併為0,   1,3合併為1  存放於mask2中
# 0 is definite background; 1 is definite foreground; 2 is probable background; 3 is probable foreground
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# np.newaxis is equal to None
# the shape is decided by the [:,:,None] (3 way)
# the None array will be one
img = img*mask2[:, :, np.newaxis]
plt.imshow(img)
plt.colorbar()
plt.show()


