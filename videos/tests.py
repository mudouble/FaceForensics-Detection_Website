# import cv2

# cap = cv2.VideoCapture('../media/in_out_videos/manipulate/585_599.avi')

# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# fourcc = int(cv2.VideoWriter_fourcc(*'H264'))

# out = cv2.VideoWriter('output.mp4', fourcc, fps, (width,  height))

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     out.write(frame)
#     # cv2.imshow('frame', frame)
#     # if cv2.waitKey(1) == ord('q'):
#     #     break

# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print("text", 2*100/123456)
# print("text", round(2.345678, 2))

# import os
# # import re
# # filepath = "E:\FF_Dection\others\output_video\manipulated_sequences"
# # filepath = re.sub(r'\\', r'/', filepath)

# # filepath = filepath + "/videos/"
# # print("ff", filepath)
# # print(os.listdir(filepath))
# filepath = "../media/in_out_videos/manipulate/Deepfakes/c40/videos"
# for i in os.listdir(filepath):
#     print("kkk", i)

# for video in os.listdir(filepath):
#     videos = "in_out_videos/manipulate/Deepfakes/c40/videos" + "/" + video
#     # print("videos", videos)
#     title = video.split(".")[0]
#     # print(title)

# #     title = ""
#     forging_method = "Deepfakes"

#     # compressed_format = "C40"
# b = Videos_Post(videos=videos, title=title, forging_method=forging_method, compressed_format="C40")
# b.save()
# import os
# from videos.models import Videos_Post

# def fun():

# filepath = "E:/FF_Detection_v1/FF_Detection/media/in_out_videos/manipulate/Deepfakes/c40/videos"
# # for i in os.listdir(filepath):
# #     print("kkk", i)

# for video in os.listdir(filepath):
#     videos = "in_out_videos/manipulate/Deepfakes/c40/videos" + "/" + video
#     title = video.split(".")[0]
#     forging_method = "Deepfakes"
#     b = Videos_Post(videos=videos, title=title, forging_method=forging_method, compressed_format="C40")
#     b.save()

# import numpy as np

# data = np.asarry([[1, 2], [3, 4], [5, 6], [7, 8]])
# print(data.shape)
