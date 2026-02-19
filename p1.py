import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: can't open webcam!")
    exit(1)

print("Presiona 'q' para salir")

# From: https://en.wikipedia.org/wiki/Relative_luminance
def calcLuminance(rgb):
    return rgb[0]*0.2126 + rgb[1]*0.7152 + rgb[2]*0.0722

def getHistogram(frame):
    frame = frameToLum(frame)
    

def frameToLum(frame):
    weights = np.array([0.299, 0.587, 0.114])
    return np.dot(frame[..., :3], weights).astype(frame.dtype)

# From: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
def correctFrame(frame):
    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist,bins = np.histogram(frame.flatten(),256,[0,256])
 
    cdf = hist.cumsum()

    cdf_m = np.ma.masked_less(cdf,5)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf_m = np.ma.filled(cdf_m,0).astype('uint8')

    corrected = cdf_m[frame]

    #cdf_orig_norm = cdf * float(hist.max()) / cdf.max()
    #cdf_mod_norm = cdf_m * float(hist.max()) / cdf_m.max()
    #plt.plot(cdf_orig_norm, color = 'b')
    #plt.hist(frame.flatten(),256,[0,256], color = 'b')
    #plt.plot(cdf_mod_norm, color = 'r')
    #plt.xlim([0,256])
    #plt.legend(('cdf','histogram'), loc = 'upper left')
    #plt.show()
    
    return corrected

def poster(frame, levels):
    levels = max(1,levels)
    x = 255//levels
    return (frame // x)*x

def nothing(x):
    pass

cv2.namedWindow("WebCam Filter")
cv2.createTrackbar("Parameter","WebCam Filter",8,32, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: can't read frame")
        break

    cv2.imshow("WebCam Original", frame)

    #frame = correctFrame(frame)
    param = cv2.getTrackbarPos("Parameter","WebCam Filter")
    frame = poster(frame, param)

    cv2.imshow("WebCam Filter", frame)

    # Wait 1ms   and read q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
