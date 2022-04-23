import time
from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import CreateView, ListView
from .forms import UploadForm
from .models import Videos_Post
import os
import re
from django.http import JsonResponse
from os.path import join
from PIL import Image as pImage
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
from videos.models_deep import model_selection
from videos.transform import xception_default_data_transforms, resnet18_default_data_transforms
from videos.transform_meso import mesonet_data_transforms
from videos.classifier import Meso4
import torchvision

num_progress = 0
frame_progress = 0
face_progress = 0
DetectImg = []
DetectPrediction = []
dictionaryProgress = {}
dictionaryProgress1 = {}
dictionaryProgress2 = {}


def index(request):
    return render(request, "index.html")


class UserVideosView(LoginRequiredMixin, ListView):
    model = Videos_Post  # xxxView的属性 model表示模型数据库
    paginate_by = 6  # 指定列表显示多少条数据
    template_name = "process_videos.html"  # 指定渲染的模板

    def get_queryset(self):
        # Videos_Post.objects.filter(id=1).delete()
        # filepath = "E:/FF_Detection_v1/FF_Detection/media/in_out_videos/original/actors/c40/videos"
        # # for i in os.listdir(filepath):
        # #     print("kkk", i)

        # for video in os.listdir(filepath):
        #     videos = "in_out_videos/original/actors/c40/videos" + "/" + video
        #     title = video.split(".")[0]
        #     forging_method = "Original actors"
        #     b = Videos_Post(videos=videos, title=title, forging_method=forging_method, compressed_format="C40")
        #     b.save()
        queryset = Videos_Post.objects.all().order_by('?')

        return queryset


class UploadVideosView(LoginRequiredMixin, CreateView):  # 基于表单处理
    model = Videos_Post
    form_class = UploadForm
    template_name = "upload.html"

    def form_valid(self, form):  # 当数据合法
        form.instance.user = self.request.user
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy("user_videoshow",
                            kwargs={"username": self.request.user.username})


def VideosInformationView(request, pk):
    models = Videos_Post.objects.get(pk=pk)
    videos = "/media/" + str(models.videos)
    dic = {
        "title": models.title,
        "videos": videos,
    }
    return render(request, "process_detail.html", dic)


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """

    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(image, modelname):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # print("just for test2", modelname)
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    if modelname == 'XceptionNet':
        preprocess = xception_default_data_transforms['test']
    elif modelname == 'MesoInceptionNet':
        preprocess = mesonet_data_transforms['test']
    elif modelname == 'ResNet18':
        preprocess = resnet18_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    print("img", preprocessed_image.shape)
    # if cuda:
    #     preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def predict_with_model(
        modelname,
        image,
        model,
        post_function=nn.Softmax(dim=1),
):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # print("just for test1", modelname)
    # Preprocess
    preprocessed_image = preprocess_image(image, modelname)

    # Model prediction
    output = model(preprocessed_image)
    # print(output)
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)  # argmax
    prediction = float(prediction.cpu().numpy())
    if modelname == "ResNet18":
        if prediction == 0: 
            prediction = 1
        else:
            prediction = 0 
    # print(prediction)

    return int(prediction), output


def funs(request, pk, m):
    # modelname = request.POST.get('models_name', )
    # pk = request.POST.get('pk', )

    time_start = time.time()
    global num_progress
    global frame_progress
    global face_progress
    global DetectImg
    global DetectPrediction
    modelname = m
    pk = pk
    print("modelname", modelname)
    print("pk", pk)
    obj = Videos_Post.objects.get(title=pk)
    videos = "/media/" + str(obj.videos)
    forging_method = obj.forging_method
    compressed_format = obj.compressed_format
    video_path = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))) + videos
    video_path = re.sub(r'\\', r'/', video_path)
    video_fn = video_path.split('/')[-1].split('.')[0] + '.mp4'
    video_file_name_only = video_path.split('/')[-1].split('.')[0]
    output_path = "E:/FF_Detection_v1/FF_Detection/media/in_out_videos/result"
    detect_path = "/media/in_out_videos/result/" + video_fn
    frame_extract = []
    face_frame = []

    if modelname == 'XceptionNet':
        model_path = "E:/FF_Detection_v1/FF_Detection/faceforensics++_models/faceforensics++_models_subset/face_detection/xception/all_c40.p"
        model, *_ = model_selection(modelname, num_out_classes=2)
        model.load_state_dict(
            torch.load(model_path,
                       map_location=torch.device(
                           "cuda" if torch.cuda.is_available() else "cpu")))
        print('Model found in {}'.format(model_path))

    elif modelname == "MesoInceptionNet":
        model_path = "E:/FF_Detection_v1/FF_Detection/faceforensics++_models/Mesonet/best.pkl"
        model = Meso4()
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path), strict=False)

    elif modelname == "ResNet18":
        model_path = "E:/FF_Detection_v1/FF_Detection/faceforensics++_models/resnet18/resnet.pth"
        model = torchvision.models.resnet18(pretrained=True)
        num = model.fc.in_features
        model.fc = nn.Linear(num, 2)

        model = torch.load(model_path) 
        # model = torch.load(model_path)

    # start to process...

    # for timess in range(12345666):
    print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = int(cv2.VideoWriter_fourcc(*'H264'))
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    face_detector = dlib.get_frontal_face_detector()
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1
    start_frame = 0
    end_frame = None
    # Frame numbers and length of output video
    print("<=== | Started Videos Splitting | ===>")
    frame_num = 0
    sequence_length = 60
    assert start_frame < num_frames - 1  # assert宏的原型定义在assert.h中，其作用是如果它的条件返回错误，则终止程序执行.
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame - start_frame)
    frames = []
    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1
        frames.append(image)
        if frame_num < start_frame:
            continue
        pbar.update(1)
    pbar.close()
    # print("frame_num", frame_num)

    for i in range(1, sequence_length + 1):
        frame = frames[i]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = pImage.fromarray(image, 'RGB')
        image_name = video_file_name_only + "_preprocessed_" + str(i) + '.png'
        image_path = "E:/FF_Detection_v1/FF_Detection/preprocess_images/" + image_name
        img.save(image_path)
        frame_extract.append(image_name)
    print("<=== | Videos Splitting Done | ===>")
    frame_progress = 1
    print("global frame_progress", frame_progress)

    print("<=== | Started Face Cropping and Predicting Each Frame | ===>")
    face_progress = 1
    print("gloabl face_progress", face_progress)
    pbar = tqdm(total=end_frame - start_frame)
    i = 0
    print("modelname", modelname)
    outs = []
    while i < num_frames:
        image = frames[i]
        height, width = image.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps,
                                     (height, width)[::-1])
        pbar.update(1)
        num_progress = i / num_frames
        # print("globa numprogress", num_progress)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y + size, x:x + size]
            if i < 60:
                image1 = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                img = pImage.fromarray(image1, 'RGB')
                image_name = video_file_name_only + "_cropped_faces_" + str(
                    i) + '.png'
                image_path = "E:/FF_Detection_v1/FF_Detection/preprocess_images/" + image_name
                img.save(image_path)
                face_frame.append(image_name)

                # Actual prediction using our model
            prediction, output = predict_with_model(
                modelname,
                cropped_face,
                model,
            )
            # ------------------------------------------------------------------

            # Text and bb
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            label = 'fake' if prediction == 1 else 'real'
            results = label
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            output_list = [
                '{0:.2f}'.format(float(x))
                for x in output.detach().cpu().numpy()[0]
            ]
            outs = output.detach().cpu().numpy()[0]
            cv2.putText(image,
                        str(output_list) + '=>' + label, (x, y + h + 30),
                        font_face, font_scale, color, thickness, 2)
            # draw box over face
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
           
            # 阈值功能
            # image1 = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            # img = pImage.fromarray(image1, 'RGB')
            # image_name = video_file_name_only + "_cropped_faces_" + str(
            #         i) + '.png'
            # image_path = "E:/FF_Detection_v1/FF_Detection/preprocess_images/" + image_name
            # img.save(image_path)
            # face_frame.append(image_name)
            # Show
        print("out", outs)
        DetectPrediction.append(outs)
        # cv2.imshow("test", image)
        # cv2.waitKey(0)
        image_name = video_file_name_only + "_detect_faces_" + str(i) + '.png'
        image_path = "E:/FF_Detection_v1/FF_Detection/preprocess_images/" + image_name
        images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      
        images = pImage.fromarray(images, 'RGB')
        images.save(image_path)
        DetectImg.append(image_name)
        cv2.waitKey(33)  # About 30 fps

        writer.write(image)
        i += 1
        # num_progress = timess * 100 / 12345666
        # print("global num_progress", num_progress)
    pbar.close()
    print("<=== | Face Cropping Each Frame Done | ===>")
    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(output_path))
    else:
        print('Input video file was empty')

    path = "/media/in_out_videos/result/" + video_fn
    time_end = time.time()
    print('totally cost', time_end - time_start)
    # path = re.sub(r'\\', r'/', path)
    return render(
        request, "process_result.html", {
            "preprocessed_images": frame_extract,
            "faces_cropped_images": face_frame,
            "resluts": results,
            "detect_path": detect_path,
            "modelname": modelname,
            "detect_videos": path,
            "compressed_format": compressed_format,
            "forging_method": forging_method,
            "title": pk
        })


def text(request, pk):
    global num_progress
    global frame_progress
    global face_progress
    num_progress = 0
    frame_progress = 0
    face_progress = 0
    obj = Videos_Post.objects.get(title=pk)
    videos = "/media/" + str(obj.videos)
    dic = {
        "title": obj.title,
        "videos": videos,
        "modles": {"XceptionNet", "MesoInceptionNet", "ResNet18"}
    }
    return render(request, "process_detect.html", dic)


def reminder2(request, num):
    print("num", num)
    t = round(num_progress*100, 2)
    dictionaryProgress[num] = t
    print("ttt", t)
    # print("show_progress----------" + str(num_progress))
    data_dict2 = {
        # "face_progress": face_progress,
        # "frame_progress": frame_progress,
        "num_progress": dictionaryProgress
    }

    return JsonResponse(data_dict2, safe=False)


def reminder1(request, num):
    print("num", num)
    dictionaryProgress1[num] = face_progress
    dictionaryProgress2[num] = frame_progress
    data_dict1 = {
        "face_progress": dictionaryProgress1,
        "frame_progress": dictionaryProgress2,
        # "num_progress": round(num_progress * 100, 2)
    }
    return JsonResponse(data_dict1, safe=False)


def threshold(request):
    yuzhi = request.POST.get('gs')
    print("yuzhi", yuzhi)
    global DetectImg
    global DetectPrediction
    # print("DetectImg", DetectImg)
    # print("DetectPrediction", DetectPrediction)
    res_pre = []
    res_img = []
    yuzhi = float(yuzhi) / 100.0
    print("hhh", yuzhi)
    index = 0
    for i in (DetectPrediction):
        # print("i", i)
        # print(yuzhi == float(i[0]))
        if yuzhi < float(i[0]) or yuzhi < float(i[1]):
            res_pre.append(index)
        index += 1
    print("res_pre", res_pre)
    for j in (res_pre):
        # print("j", j)
        # print("img", DetectImg[j])
        res_img.append(DetectImg[j])

    return render(request, "models_details.html", {"totallen": len(DetectImg), "threshold": res_img, "p": "frames search for you", "len": len(res_pre), "yuzhi": yuzhi, "detectImg": DetectImg})


def ModelsDetailView(request):
    return render(request, "models_details.html", {"detectImg": DetectImg, "totallen": len(DetectImg)})
    
