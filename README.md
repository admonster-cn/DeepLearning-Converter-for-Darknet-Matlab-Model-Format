# Darknet Importer and Exporter
The importer can import all the seriesNetworks in the darknet and some simple DAGnetworks. The exporter can export all the seriesNetworks and some of the backbone networks. In addition to importing the deep neural network, the importer can obtain the feature map size of the network, the number of parameters, and the computational power FLOPs. For yolov2, yolov3 can also import a number of previous modules for later access to the yolo layer. This program requires Matlab2019a version and above, no other dependencies.

***
<p>
目前测试能导入和导出官网（https://github.com/pjreddie/darknet/tree/master/cfg)以下网络,其他resnet* ，yolo*可导入部分网络层
alexnet.cfg
cifar.cfg
 darknet.cfg 
darknet19.cfg
darknet19_448.cfg
darknet53.cfg 
darknet9000.cfg
 densenet201.cfg 
extraction.cfg
 extraction.conv.cfg
 extraction22k.cfg tiny.cfg
tiny.cfg
能够计算导入网络的特征图大小，FLOPs，参数个数。
1、示例一：导入模型示例<br>
从https://pjreddie.com/darknet/imagenet/ 找到Pre_Trained Models的下载链接并点击下载，比如使用darknet19
  
```
cfg_file = 'G:\download\darknet-master\cfg\darknet19.cfg';
weight_file = 'G:\download\darknet-master\cfg\darknet19.weights';
[mynet,hyperParams,numsNetParams,FLOPs] = importDarknetNetwork(cfg_file,weight_file);
analyzeNetwork(mynet)

% 获取classesNames
fid = fopen('imagenet_shortnames_list.txt','r');
cls = textscan(fid, '%s', 'Delimiter',{'\n'});
classesNames = cls{1};
fclose(fid);

%  预测top5
img = imread('peppers.png');
input_size = mynet.Layers(1).InputSize;
img =im2single(imresize(img,input_size(1:2)));
scores = predict(mynet,img);
rank_k = 5;
[max_scores,ids] = maxk(scores,rank_k,2);
for i = 1:rank_k
    predictLabel = classesNames{ids(i)};
    predictScore = max_scores(i);
    fprintf('top%d, predictLabel:%-20s,predictScore:%.2f\n',i,predictLabel,predictScore);
end
```

2、示例二：导入部分层和权重<br>
```
cutoffModule = 15;
[lgraphWeight,hyperParams,numsNetParams,FLOPs] = importDarknetWeights(cfg_file,weight_file,cutoffModule);
analyzeNetwork(lgraphWeight)
```

3、示例三：导出mynet为cfg,weights 文件
```
hyperParams = [];
if isempty(hyperParams)
    hyperParams = struct('batch',64,...
        'subdivisiions',1,...
        'height',mynet.Layers(1).InputSize(1),...
        'width',mynet.Layers(1).InputSize(2),...
        'channels',mynet.Layers(1).InputSize(3),...
        'momentum',0.9,...
        'max_crop',256,...
        'learning_rate',0.01,...
        'policy','poly',...
        'power',4,...
        'max_batches',600000,...
        'angle',7,...
        'saturation',0.75,...
        'exposure',0.75,...
        'aspect',0.75);
end
cfgfile = 'exportC.cfg';
weightfile = 'exportW.weights';
exportDarkNetNetwork(mynet,hyperParams,cfgfile,weightfile)
```

4、示例四：导入部分层示例
比如使用yolov3-tiny.cfg
```
cfgfile = 'G:\download\darknet-master\cfg\yolov3-tiny.cfg';
cutoffModule = 17;
[lgraphLayer,hyperParams,numsNetParams,FLOPs,moduleTypeList,moduleInfoList,layerToModuleIndex] = importDarkNetLayers(cfgfile,cutoffModule);
analyzeNetwork(lgraphLayer)
```
