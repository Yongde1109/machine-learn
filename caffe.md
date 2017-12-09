# caffe
1. translate pretrained caffe model into caffe2
    
    ```bash
    python -m caffe2.python.caffe_translator deploy.prototxt pretrained.caffemodel
    ```
2. load pretrained model with cpu
    
    ```python
    from caffe2.proto import caffe2_pb2
    import numpy as np
    import skimage.io
    import skimage.transform
    from caffe2.python import core, workspace, models

    def crop_center(img,cropx,cropy):
        y,x,c = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        return img[starty:starty+cropy,startx:startx+cropx]

    def rescale(img, input_height, input_width):
        print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
        print("Model's input shape is %dx%d") % (input_height, input_width)
        aspect = img.shape[1]/float(img.shape[0])
        print("Orginal aspect ratio: " + str(aspect))
        if(aspect>1):
            # landscape orientation - wide image
            res = int(aspect * input_height)
            imgScaled = skimage.transform.resize(img, (input_width, res))
        if(aspect<1):
            # portrait orientation - tall image
            res = int(input_width/aspect)
            imgScaled = skimage.transform.resize(img, (res, input_height))
        if(aspect == 1):
            imgScaled = skimage.transform.resize(img, (input_width, input_height))
    
        print("New image shape:" + str(imgScaled.shape) + " in HWC")
        return imgScaled
        
    INPUT_IMAGE_SIZE = 227
    MEAN_FILE = '/mnt/tmp-img/ml/ILGnet/mean/AVA2_mean.npy'

    IMAGE_LOCATION = '/mnt/tmp-img/ml/ava/AVA_dataset/test/images/104604.jpg'

    mean = np.load(MEAN_FILE).mean(1).mean(1)
    mean = mean[:, np.newaxis, np.newaxis]

    img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
    img = rescale(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    img = crop_center(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)

    img = img.swapaxes(1, 2).swapaxes(0, 1)

    img = img[(2, 1, 0), :, :]
    img = img * 255 - mean
    # add batch size
    img = img[np.newaxis, :, :, :].astype(np.float32)

    INIT_NET = 'init_net.pb'
    PREDICT_NET = 'predict_net.pb'

    with open(INIT_NET) as f:
        init_net = f.read()
    with open(PREDICT_NET) as f:
        predict_net = f.read()

    print workspace.has_gpu_support

    p = workspace.Predictor(init_net, predict_net)

    # run the net and return prediction
    results = p.run([img])

    print results
    ```
3. load pretrained model with gpu 
    
    ```python
    INIT_NET = 'init_net.pb'
    PREDICT_NET = 'predict_net.pb'

    workspace.ResetWorkspace();
    device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0)

    init_def = caffe2_pb2.NetDef()
    with open(INIT_NET, 'rb') as f:
        init_def.ParseFromString(f.read())
        init_def.device_option.CopyFrom(device_opts)
        workspace.RunNetOnce(init_def.SerializeToString())

    net_def = caffe2_pb2.NetDef()
    with open(PREDICT_NET, 'rb') as f:
        net_def.ParseFromString(f.read())
        net_def.device_option.CopyFrom(device_opts)
        workspace.CreateNet(net_def.SerializeToString())
    
    name = net_def.name
    out_name = net_def.external_output[-1];
    in_name = net_def.external_input[0]

    #%%
    workspace.FeedBlob(in_name, img, device_opts)
    print ('Running net..."' + name + '"' )
    workspace.RunNet(name, 1)
    results = workspace.FetchBlob(out_name)

    print results
    ```
4. to be continued

