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

def lerp(x,y,a):
    return x*(1-a)+y*a

# From: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
def correctFrame(frame):
    #cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist,bins = np.histogram(frame.flatten(),256,[0,256])
 
    cdf = hist.cumsum()

    # Mask all pixels with value < 5 
    cdf_m = np.ma.masked_less(cdf,5)
    # Normalize (MinMax)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    # Replace masked pixels with 0
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

def poster(frame, param):
    levels = int(np.pow(2,np.trunc(lerp(1,5,param))))
    print(levels)
    x = 255//levels
    return (frame // x)*x

def alien(frame, param, color):
    # Format to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define color treshold
    # https://stackoverflow.com/questions/8753833/exact-skin-color-hsv-range
    x = lerp(0,50,param)
    y = lerp(23,68,param)
    lower_skin = np.array([0, x, y], dtype=np.uint8)
    upper_skin = np.array([x, 255, 255], dtype=np.uint8)
    # Mask in range colors
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # Color in masked frame
    frame[mask > 0] = color
    return frame

def barrelCusion(frame, param):
    k1 = (param*2-1) * 1e-5
    k2 = k1*1e-5
    map_x = x_grid + xd_rel * (k1 * r2 + k2 * r4)
    map_y = y_grid + yd_rel * (k1 * r2 + k2 * r4)

    # Apply remap
    return cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)

# Based from https://github.com/VictorOrrios/victorr.me/blob/main/src/lib/components/backgrounds/filters/BayesDither/fragment.glsl
def bayesDither(frame, param):
    BAYER_4x4 = (1/16) * np.array([
        [0,  8,  2, 10],
        [12, 4, 14, 6 ],
        [3, 11, 1, 9 ],
        [15, 7, 13, 5 ]
    ], dtype=np.float32)

    h, w, _ = frame.shape

    # Get liminance for each pixel
    lum = (0.2126*frame[:,:,2] +
           0.7152*frame[:,:,1] +
           0.0722*frame[:,:,0]) / 255.0

    # Tile bayer matrix to frame extent
    tiled = np.tile(BAYER_4x4, (h//4 + 1, w//4 + 1))[:h, :w]

    # Apply threshold
    mask = lum * param*2 > tiled

    # Copy pixels where mask meets threshold
    out = np.zeros_like(frame)
    out[mask] = frame[mask]

    return out

def pixelize(frame, param):
    size = 2**int(lerp(1,6,param))
    h, w, _ = frame.shape
    temp = cv2.resize(frame, (int(w/size),int(h/size)), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(temp, (w,h), interpolation=cv2.INTER_NEAREST)
    
def polkaDots(frame, param):
    h, w, _ = frame.shape
    size = 2**int(lerp(1,6,param))

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    frame[yy,xx] = frame[(yy//size)*size,(xx//size)*size]
    yy_rel = yy - (yy // size) * size
    xx_rel = xx - (xx // size) * size
    mask = np.sqrt((yy_rel - size/2)**2 + (xx_rel - size/2)**2) > size/2.5

    frame[mask] = [0,0,0]

    return frame
    

def nothing(x):
    pass

cv2.namedWindow("WebCam Filter")
cv2.createTrackbar("Parameter","WebCam Filter",1,100, nothing)

# Pre-calculate coordinates for the map
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
xcen = frame_width / 2
ycen = frame_height / 2
x_grid, y_grid = np.meshgrid(np.arange(frame_width), np.arange(frame_height))
xd_rel = x_grid - xcen
yd_rel = y_grid - ycen
r2 = xd_rel**2 + yd_rel**2
r4 = r2**2

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: can't read frame")
        break

    #cv2.imshow("WebCam Original", frame)

    param = cv2.getTrackbarPos("Parameter","WebCam Filter")
    param /= 100.0
    #frame = correctFrame(frame)
    #frame = poster(frame, param)
    #frame = alien(frame, param, [255, 0, 0]) # Blue skin
    #frame = alien(frame, param, [0, 255, 0]) # Green skin
    #frame = alien(frame, param, [0, 0, 255]) # Red skin
    #frame = barrelCusion(frame, param)
    #frame = bayesDither(frame, param)
    #frame = pixelize(frame,param)
    frame = polkaDots(frame,param)

    
    cv2.imshow("WebCam Filter", frame)

    # Wait 1ms   and read q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
