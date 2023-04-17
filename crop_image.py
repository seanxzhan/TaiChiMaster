import cv2

n = 5
fp = f'data/demo/{n}.png'
frame = cv2.imread(fp)
print(frame.shape)
# img = img[200:1200, 100:2600]
scale_percent = 40 * 1.238 # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
print(dim)
frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
frame = frame[100:580, 220:860]
# frame = frame[60:540, 220:860]
# frame
print(frame.shape)
cv2.imshow('fig', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(f'data/demo/{n}_resized.png', frame)