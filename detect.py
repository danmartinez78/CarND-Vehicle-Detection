import skvideo.io
from funcs import *

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

# initialize heat map
image = mpimg.imread('./test_images/test1.jpg')
heat = np.zeros_like(image[:,:,0]).astype(np.float)
# open stream
path = 'project_video.mp4'
save = True
debug = False
stream = skvideo.io.vread(path)
cv2.waitKey(500)
print("got stream")
writer = skvideo.io.FFmpegWriter("result.mp4", outputdict={'-r': '10'})
for frame in stream:
    output, heat = process_frame(frame, heat, y_start, y_stop, scales, svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins)
    if debug:
        cv2.imshow("debug", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    if save:
        # write to video
        writer.writeFrame(output)
    cv2.waitKey(1)

stream.release()
cv2.destroyAllWindows()
writer.close()
