from functions import *
import pickle
import matplotlib.pyplot as plt
import skvideo.io
import matplotlib.image as mpimg

fig = plt.figure(figsize=(30, 20))
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.subplot(1,2,1)
vehicle = plt.imread('data/vehicles/GTI_Far/image0024.png')
plt.imshow(vehicle)
plt.title('Vehicle', fontsize = '50')
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
non_vehicle = plt.imread('data/non-vehicles/GTI/image24.png')
plt.imshow(non_vehicle)
plt.title('Non-Vehicle', fontsize = '50')
plt.xticks([])
plt.yticks([])
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
fig.tight_layout()
fig.savefig('output_images/car_not_car.png', bbox_inches='tight')

plt.clf()
plt.subplot(1,2,1)
plt.imshow(vehicle)
plt.title('Vehicle', fontsize = '50')
plt.xticks([], fontsize=25)
plt.yticks([], fontsize=25)

plt.subplot(1,2,2)
features, hog = get_hog_features(vehicle[:,:,0], orient=8, pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=True)
plt.imshow(hog, cmap='gray')
plt.title('HOG Image', fontsize = '50')
plt.xticks([], fontsize=25)
plt.yticks([], fontsize=25)
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
fig.tight_layout()
fig.savefig('output_images/HOG.png', bbox_inches='tight')

dist_pickle = pickle.load(open("vehicle_svc.p", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["X_scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
hog_channel = dist_pickle["hog_channel"]
spatial_feat = dist_pickle["spatial_feat"]
hist_feat = dist_pickle["hist_feat"]
hog_feat = dist_pickle["hog_feat"]
color_space = dist_pickle["color_space"]

y_start = [400, 400]
y_stop = [500,  656]
scales = [1.25, 1.5]

img = mpimg.imread('test_images/test1.jpg')

heat = np.zeros_like(img[:,:,0]).astype(np.float)
out_img, heat = process_frame(img, heat, y_start, y_stop, scales, svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins)
fig = plt.figure(figsize=(30, 20))
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.subplot(121)
plt.imshow(out_img)
plt.title('Car Positions', fontsize = '50')
plt.subplot(122)
plt.imshow(heat, cmap='hot')
plt.title('Heat Map', fontsize = '50')
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
fig.tight_layout()
fig.savefig('output_images/1.png', bbox_inches='tight')

img = mpimg.imread('test_images/test2.jpg')

heat = np.zeros_like(img[:,:,0]).astype(np.float)
out_img, heat = process_frame(img, heat, y_start, y_stop, scales, svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins)
fig = plt.figure(figsize=(30, 20))
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.subplot(121)
plt.imshow(out_img)
plt.title('Car Positions', fontsize = '50')
plt.subplot(122)
plt.imshow(heat, cmap='hot')
plt.title('Heat Map', fontsize = '50')
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
fig.tight_layout()
fig.savefig('output_images/2.png', bbox_inches='tight')

img = mpimg.imread('test_images/test3.jpg')

heat = np.zeros_like(img[:,:,0]).astype(np.float)
out_img, heat = process_frame(img, heat, y_start, y_stop, scales, svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins)
fig = plt.figure(figsize=(30, 20))
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.subplot(121)
plt.imshow(out_img)
plt.title('Car Positions', fontsize = '50')
plt.subplot(122)
plt.imshow(heat, cmap='hot')
plt.title('Heat Map', fontsize = '50')
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
fig.tight_layout()
fig.savefig('output_images/3.png', bbox_inches='tight')

img = mpimg.imread('test_images/test4.jpg')

heat = np.zeros_like(img[:,:,0]).astype(np.float)
out_img, heat = process_frame(img, heat, y_start, y_stop, scales, svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins)
fig = plt.figure(figsize=(30, 20))
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.subplot(121)
plt.imshow(out_img)
plt.title('Car Positions', fontsize = '50')
plt.subplot(122)
plt.imshow(heat, cmap='hot')
plt.title('Heat Map', fontsize = '50')
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
fig.tight_layout()
fig.savefig('output_images/4.png', bbox_inches='tight')


colors = [(0,255,0),(255,0,0)]
img = mpimg.imread('test_images/test3.jpg')
for start, stop, scale, color in zip(y_start, y_stop, scales, colors):
    img = draw_multi_scale_windows(img, start, stop, scale, pix_per_cell, orient, cell_per_block, color)
cv2.imwrite('output_images/windows.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# initialize heat map
image = mpimg.imread('./test_images/test1.jpg')
heat = np.zeros_like(image[:,:,0]).astype(np.float)
# open stream
path = 'test_video.mp4'
stream = skvideo.io.vread(path)
cv2.waitKey(500)
print("got stream")
writer = skvideo.io.FFmpegWriter("result.mp4", outputdict={'-r': '10'})
i = 0
for frame in stream:
    output, heat, box_image, labels = process_frame_for_report(frame, heat, y_start, y_stop, scales, svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins)
    fig = plt.figure(figsize=(30, 20))
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.subplot(121)
    plt.imshow(box_image)
    plt.title('Car Positions', fontsize = '50')
    plt.subplot(122)
    plt.imshow(heat, cmap='hot')
    plt.title('Heat Map', fontsize = '50')
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.savefig('output_images/frame_%s.png' % i, bbox_inches='tight')
    plt.close("all")
    i += 1
    cv2.imshow("debug", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    if i > 6:
        l = plt.figure(figsize=(30, 20))
        print(labels[1], 'cars found')
        plt.imshow(labels[0], cmap='gray')
        l.savefig('output_images/labels.png', bbox_inches='tight')
        cv2.imwrite('output_images/final_bb.png', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        break
stream.release()
cv2.destroyAllWindows()
writer.close()