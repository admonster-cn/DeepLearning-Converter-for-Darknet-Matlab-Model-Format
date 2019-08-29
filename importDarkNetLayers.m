function [lgraph,hyperParams,numsNetParams,FLOPs,moduleTypeList,moduleInfoList,layerToModuleIndex] = importDarkNetLayers(cfgfile,varargin)
% importDarkNetLayers ���ܣ���darknet��cfgfile����Ϊmatlab��lgraph
% ���룺cfgfile, (��ѡ��)�ַ�������ָ����cfg��׺��ģ�������ļ�
%      cutoffModule,(��ѡ��)1*1����������ָ������darknetǰcutoffModule��module����1Ϊbase��������û�и�����������������
% �����lgraph�� matlab���ѧϰģ��ͼ��Ŀǰֻ֧��series network����DAGnetwork
%      hyperParams,�ṹ�壬���������ļ�
%      numsNetParams,Ȩ�ز�������
%      FLOPs�� ģ�ͼ�����
%      moduleTypeList,cell array���ͣ�����ÿ���ַ������洢module������
%      moduleInfoList��cell array���ͣ�����ÿ��struct�洢module����Ϣ�� ��������ÿ���ṹ��洢cfg�е������⣬������洢channels��mapsize��������
%      layerToModuleIndex,
%      n*1���������飬lgraph��Layers��Ϊmodule����������1��ʼ��nΪLayers�ĳ��ȴ�С
% ע�⣺1���ʺ�2019a�汾������
%      2��leaky��ֵĿǰȡ��0.1
%      3�����ĳ��module����bn�㣬��conv��biasΪ0����Ϊdarknet�����ִ洢��ʽ
%      4�������뵽yolo�㣬�˳����룬��ʱ��֧��yolo������Ĳ㣬��Ϊyolov3�ٷ�����֧��
%      5��shortcut��routeֻ֧������ڵ���������2����������һ����֧��·û��layer
%      6��darknet weights����˳������ΪBN��offset,scale,mean,variance,Conv���bias,weights
%      ����ͼ���output Size = (Input Size �C ((Filter Size �C 1)*Dilation Factor + 1) + 2*Padding)/Stride + 1
% �ο���1���ٷ��ĵ���Specify Layers of Convolutional Neural Network
%      2��https://www.zhihu.com/question/65305385
%       3��https://github.com/ultralytics/yolov3/blob/master/models.py
% cuixingxing150@gmail.com
% 2019.8.19
% �޸���2019.8.29������relu6֧��
%
minArgs=1;
maxArgs=2;
narginchk(minArgs,maxArgs)
% fprintf('Received 1 required and %d optional inputs\n', length(varargin));

%% init
numsNetParams = 0;FLOPs = 0;

%% ��������cfg�ļ�
fid = fopen(cfgfile,'r');
cfg = textscan(fid, '%s', 'Delimiter',{'   '});
fclose(fid);
cfg = cfg{1};
TF = startsWith(cfg,'#');
cfg(TF) = [];

%% ����module��info��Ϣ�㼯
TF_layer = startsWith(cfg,'[');
moduleTypeList = cfg(TF_layer);
nums_Module = length(moduleTypeList);
moduleInfoList = cell(nums_Module,1);%

%% ��ȡ���������ļ�
indexs = find(TF_layer);
for i = 1:nums_Module
    if i == nums_Module
        moduleInfo = cfg(indexs(i)+1:end,:);
    else
        moduleInfo = cfg(indexs(i)+1:indexs(i+1)-1,:);
    end
    if ~isempty(moduleInfo)
        moduleInfo = strip(split(moduleInfo,'='));
        moduleInfo = reshape(moduleInfo,[],2);
        structArray = cell2struct(moduleInfo, moduleInfo(:,1), 1);
        moduleStruct = structArray(2);
        moduleInfoList{i} = moduleStruct;
    else
        moduleInfoList{i} = [];
    end
end

%% cutoff
if ~isempty(varargin)
    nums_Module = varargin{1};
    moduleTypeList(nums_Module+1:end) = [];
    moduleInfoList(nums_Module+1:end) = [];
end

%% ��������ṹͼ
lgraph = layerGraph();hyperParams = struct();
moduleLayers = []; lastModuleNames = cell(nums_Module,1);layerToModuleIndex=[];
for i = 1:nums_Module
    currentModuleType = moduleTypeList{i};
    currentModuleInfo = moduleInfoList{i};
    switch currentModuleType
        case '[net]'
            hyperParams = currentModuleInfo;
            if all(isfield(currentModuleInfo,{'height','width','channels'}))
                height = str2double(currentModuleInfo.height);
                width =  str2double(currentModuleInfo.width);
                channels = str2double(currentModuleInfo.channels);
                imageInputSize = [height,width,channels];
                moduleInfoList{i}.channels = channels; % �������������������������FLOPs
                moduleInfoList{i}.mapSize = [height ,width];% �������FLOPs�����һ��ػ���С
            else
                error('[net] require height, width,channels parameters in cfg file!');
            end
            input_layer = imageInputLayer(imageInputSize,'Normalization','none',...
                'Name','input_1');
            moduleLayers = input_layer;
            lgraph = addLayers(lgraph,moduleLayers);
        case '[convolutional]'
            % ���conv��
            moduleLayers = [];conv_layer = [];bn_layer = [];relu_layer = [];
            nums_p=numsNetParams;% ���㵱ǰmodule��Ȩ�ظ���
            filterSize = str2double(currentModuleInfo.size);
            numFilters = str2double(currentModuleInfo.filters);
            stride = str2double(currentModuleInfo.stride);
            pad = str2double(currentModuleInfo.pad);
            if stride==1
                pad ='same';
            end
            channels_in = moduleInfoList{i-1}.channels;
            if isfield(currentModuleInfo,'groups')
                numGroups = str2double(currentModuleInfo.groups);
                numFiltersPerGroup_out = numFilters/numGroups;
                conv_layer = groupedConvolution2dLayer(filterSize,numFiltersPerGroup_out,numGroups,...
                    'Name',['dw_conv_',num2str(i)],'Stride',stride,...
                    'Padding',pad);
                numsNetParams = numsNetParams +(filterSize*filterSize*channels_in/numGroups*numFiltersPerGroup_out*numGroups);
                numsNetParams = numsNetParams +numFiltersPerGroup_out*numGroups; % bias
            else
                conv_layer = convolution2dLayer(filterSize,numFilters,'Name',['conv_',num2str(i)],...
                    'Stride',stride,'Padding',pad);
                numsNetParams = numsNetParams +(filterSize*filterSize*channels_in*numFilters);% weights
                numsNetParams = numsNetParams +numFilters; % bias
            end
            moduleInfoList{i}.channels =numFilters;
            if ischar(pad)
                moduleInfoList{i}.mapSize = moduleInfoList{i-1}.mapSize;
            else
                dilationF=1;
                moduleInfoList{i}.mapSize = floor((moduleInfoList{i-1}.mapSize-((filterSize-1)*dilationF +1)+2*pad)/stride+1);
            end
            
            % ���BN��
            if isfield(currentModuleInfo,'batch_normalize')
                bn_layer = batchNormalizationLayer('Name',['bn_',num2str(i)]);
                numsNetParams = numsNetParams +numFilters*4;% offset,scale,mean,variance
            end
            FLOPs_perConv = prod(moduleInfoList{i}.mapSize)*(numsNetParams-nums_p);
            FLOPs = FLOPs+FLOPs_perConv;
            fprintf('This module No:%2d [convolutional],have #params:%-10d,FLops:%-12d,feature map size:(%3d*%3d)\n',...
                i,numsNetParams-nums_p,FLOPs_perConv,moduleInfoList{i}.mapSize);
            
            % ���relu��
            if strcmp(currentModuleInfo.activation,'relu')
                relu_layer = reluLayer('Name',['relu_',num2str(i)]);
            elseif strcmp(currentModuleInfo.activation,'relu6')
                relu_layer = clippedReluLayer(6,'Name',['clipRelu_',num2str(i)]);
            elseif strcmp(currentModuleInfo.activation,'leaky')
                relu_layer = leakyReluLayer(0.1,'Name',['leaky_',num2str(i)]);
            end
            moduleLayers = [conv_layer;bn_layer;relu_layer];
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{i-1},moduleLayers(1).Name);
        case '[shortcut]'
            moduleLayers = [];add_layer=[];relu_layer = [];
            connectID = strip(split(currentModuleInfo.from,','));% connectIDΪcell
            if length(connectID)>2
                error('unsupport more than 2 inputs');
            end
             if length(connectID)==1
                module_idx1 = i-1;
                temp = str2double(connectID);
                module_idx2 = getModuleIdx(i,temp);
            else
                temp1 = str2double(connectID(1));temp2 = str2double(connectID(2));
                module_idx1 = getModuleIdx(i,temp1);
                module_idx2 = getModuleIdx(i,temp2);
             end
            add_layer = additionLayer(2,'Name',['add_',num2str(i)]);
            moduleInfoList{i}.channels =moduleInfoList{i-1}.channels;
            moduleInfoList{i}.mapSize = moduleInfoList{i-1}.mapSize;
            % ���relu��
            if strcmp(currentModuleInfo.activation,'relu')
                relu_layer = reluLayer('Name',['relu_',num2str(i)]);
            elseif strcmp(currentModuleInfo.activation,'relu6')
                relu_layer = clippedReluLayer(6,'Name',['clipRelu_',num2str(i)]);
            elseif strcmp(currentModuleInfo.activation,'leaky')
                relu_layer = leakyReluLayer('Name',['leaky_',num2str(i)]);
            end
            moduleLayers = [add_layer;relu_layer];
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{module_idx1},[moduleLayers(1).Name,'/in1']);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{module_idx2},[moduleLayers(1).Name,'/in2']);
        case '[route]'
            moduleLayers = [];depth_layer = [];relu_layer = [];
            connectID = strip(split(currentModuleInfo.layers,','));
            if length(connectID)>2
                error('unsupport more than 2 inputs');
            end
            if length(connectID)==1
                module_idx1 = i-1;
                temp = str2double(connectID);
                module_idx2 = getModuleIdx(i,temp);
            else
                temp1 = str2double(connectID(1));temp2 = str2double(connectID(2));
                module_idx1 = getModuleIdx(i,temp1);
                module_idx2 = getModuleIdx(i,temp2);
            end
            depth_layer = depthConcatenationLayer(2,'Name',['concat_',num2str(i)]);
            moduleInfoList{i}.channels = moduleInfoList{module_idx1}.channels+moduleInfoList{module_idx2}.channels;
            moduleInfoList{i}.mapSize = moduleInfoList{i-1}.mapSize;
            % ���relu��
            if isfield(currentModuleInfo,'activation')
                if strcmp(currentModuleInfo.activation,'relu')
                    relu_layer = reluLayer('Name',['relu_',num2str(i)]);
                elseif strcmp(currentModuleInfo.activation,'relu6')
                    relu_layer = clippedReluLayer(6,'Name',['clipRelu_',num2str(i)]);
                elseif strcmp(currentModuleInfo.activation,'leaky')
                    relu_layer = leakyReluLayer('Name',['leaky_',num2str(i)]);
                end
            end
            moduleLayers = [depth_layer;relu_layer];
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{module_idx1},[moduleLayers(1).Name,'/in1']);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{module_idx2},[moduleLayers(1).Name,'/in2']);
        case '[avgpool]'
            moduleLayers = [];avg_layer = [];
            poolsize = moduleInfoList{i-1}.mapSize;
            pad =0;stride=1;
            if isempty(currentModuleInfo) % Ϊ��ʱ���Զ��ƶϴ�С,��Ϊ��һ������ͼ��С
                avg_layer = averagePooling2dLayer(poolsize,'Padding',pad,...
                    'Stride',stride,'Name',['avgPool_',num2str(i)]);
            else
                poolsize = str2double(currentModuleInfo.size);
                stride = str2double(currentModuleInfo.stride);
                pad = 'same'; % ȷ��strideΪ1ʱ������ͼ��С����
                 if isfield(currentModuleInfo,'padding')
                    pad = str2double(currentModuleInfo.padding);
                end
                avg_layer = averagePooling2dLayer(poolsize,'Padding',pad,...
                    'Stride',stride,'Name',['avgPool_',num2str(i)]);
            end
            moduleInfoList{i}.channels = moduleInfoList{i-1}.channels;
            if ischar(pad)&&stride==1
                 moduleInfoList{i}.mapSize =  moduleInfoList{i-1}.mapSize;
            elseif ischar(pad)
                moduleInfoList{i}.mapSize = ceil(moduleInfoList{i-1}.mapSize/stride);
            else
                moduleInfoList{i}.mapSize = floor((moduleInfoList{i-1}.mapSize-poolsize+2*pad)/stride+1);
            end    
            
            moduleLayers= avg_layer;
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{i-1},moduleLayers(1).Name);
        case '[maxpool]'
            moduleLayers = [];maxp_layer = [];
            poolsize = moduleInfoList{i-1}.mapSize;
            pad =0;stride=1;
            if isempty(currentModuleInfo) % Ϊ��ʱ���Զ��ƶϴ�С,��Ϊ��һ������ͼ��С
                maxp_layer = maxPooling2dLayer(poolsize,'Padding',pad,...
                    'Stride',stride,'Name',['avgPool_',num2str(i)]);
            else
                poolsize = str2double(currentModuleInfo.size);
                stride = str2double(currentModuleInfo.stride);
                pad = 'same'; % ȷ��strideΪ1ʱ������ͼ��С����
                if isfield(currentModuleInfo,'padding')
                    pad = str2double(currentModuleInfo.padding);
                end
                maxp_layer = maxPooling2dLayer(poolsize,'Padding',pad,...
                    'Stride',stride,'Name',['maxPool_',num2str(i)]);
            end
            moduleInfoList{i}.channels = moduleInfoList{i-1}.channels;
            if ischar(pad)&&stride==1
                 moduleInfoList{i}.mapSize =  moduleInfoList{i-1}.mapSize;
            elseif ischar(pad)
                moduleInfoList{i}.mapSize = ceil(moduleInfoList{i-1}.mapSize/stride);
            else
                moduleInfoList{i}.mapSize = floor((moduleInfoList{i-1}.mapSize-poolsize+2*pad)/stride+1);
            end    
            
            moduleLayers= maxp_layer;
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{i-1},moduleLayers(1).Name);
        case '[dropout]'
            moduleLayers = [];drop_layer = [];
            probability = str2double(currentModuleInfo.probability);
            drop_layer = dropoutLayer(probability,'Name',['drop_',num2str(i)]);
            moduleInfoList{i}.channels = moduleInfoList{i-1}.channels;
            moduleInfoList{i}.mapSize = moduleInfoList{i-1}.mapSize;
            
            moduleLayers= drop_layer;
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{i-1},moduleLayers(1).Name);
        case '[connected]' % ����ͨ�����������������С�Ƿ�̶�����֤ȫ���Ӳ�����ɳ�;�����ʱ�����ǽ�BN
            moduleLayers = [];connected_layer = [];relu_layer = [];
            output = str2double(currentModuleInfo.output);
            connected_layer = fullyConnectedLayer(output,'Name',['fullyCon_',num2str(i)]);
            moduleInfoList{i}.channels = output;
            moduleInfoList{i}.mapSize = [1,1];
            
            % ���relu��
            if strcmp(currentModuleInfo.activation,'relu')
                relu_layer = reluLayer('Name',['relu_',num2str(i)]);
            elseif strcmp(currentModuleInfo.activation,'relu6')
                relu_layer = clippedReluLayer(6,'Name',['clipRelu_',num2str(i)]);
            elseif strcmp(currentModuleInfo.activation,'leaky')
                relu_layer = leakyReluLayer(0.1,'Name',['leaky_',num2str(i)]);
            end

            moduleLayers= [connected_layer;relu_layer];
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{i-1},moduleLayers(1).Name);
        case '[softmax]'
            moduleLayers = [];soft_layer = [];
            soft_layer = softmaxLayer('Name',['softmax_',num2str(i)]);
            moduleInfoList{i}.channels = moduleInfoList{i-1}.channels;
            moduleInfoList{i}.mapSize = moduleInfoList{i-1}.mapSize;
            
            moduleLayers= soft_layer;
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{i-1},moduleLayers(1).Name);
        case '[cost]'
            moduleLayers = [];clss_layer = [];
            clss_layer = classificationLayer('Name',['clss_',num2str(i)]);
            moduleInfoList{i}.channels = moduleInfoList{i-1}.channels;
            moduleInfoList{i}.mapSize = moduleInfoList{i-1}.mapSize;
            
            moduleLayers= clss_layer;
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{i-1},moduleLayers(1).Name);
        otherwise
            error("we currently can't support this layer: "+currentModuleType);
    end
    lastModuleNames{i} = moduleLayers(end).Name;
    layerToModuleIndex = [layerToModuleIndex;i*ones(length(moduleLayers),1)];
end

    function module_idx = getModuleIdx(current_ind,cfg_value)
        % route,����shortcut��ת��Ϊ��1Ϊ��ʼ�����ı���ֵ
        % ���룺current_ind�����뵽��ǰ��module����������([net]��1Ϊ��ʼֵ),darknet����[net]Ϊ0��ʼֵ
        %      cfg_value��shortcut���fromֵ����route��layers��ĳһ��ֵ
        % �����module_idx�����ӵ���һ��module������ֵ��������,��[net]Ϊ��ʼ����1��
        %
        % cuixingxing150@gmail.com
        % 2019.8.19
        %
        if cfg_value<0
            module_idx = current_ind+cfg_value;
        else
            module_idx = 1+cfg_value;
        end
    end % end of getModuleIdx

end % end of importDarknetLayers
