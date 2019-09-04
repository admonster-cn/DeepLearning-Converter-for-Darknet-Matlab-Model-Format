function exportDarkNetNetwork(net,hyperParams,cfgfileName,weightfileName,varargin)
% EXPORTDARKETNetNetwork ���ܣ���matlab���ѧϰģ�͵���Ϊdarknet��ģ��weights�ļ�
% ���룺net�� matlab���ѧϰģ�ͣ�Ŀǰ��ֻ֧��series network
%      hyperParams,�ṹ�壬���������ļ�
%      varargin ��cutoffModule,(��ѡ��)1*1����������ָ������darknetǰcutoffModule��module����cfg�ļ��е�һ����[net]��ʼ��moduleΪ0��ʼ�ļ�����û�и����򵼳���������
% �����
%      cfgfile, ָ����cfg��׺��ģ�������ļ�
%      weightfile,�ƶ���Ӧ��weightsȨ���ļ�
%
% ע�⣺1��relu6��relu��������棬��Ϊclip relu��֪��darknet�Ƿ�ʵ��
%      2��matlab��module��[net]Ϊ1��ʼ��������darknet��cfg��[convolutional]��һ��Ϊ0��ʼ����
%      3��һ��moduleΪ������BN,Activation��ʼ�Ĳ�
% cuixingxing150@gmail.com
% 2019.8.22
% 2019.8.29�޸ģ�֧�ֵ���relu6
% 2019.9.4�޸ģ���ԭ����darknet��[net]Ϊ0��ʼ��������Ϊ��cfg�ļ��е�һ����[net]��ʼ��moduleΪ0��ʼ�ļ���������
%
minArgs=4;
maxArgs=5;
narginchk(minArgs,maxArgs)

%% init
moduleTypeList = []; % cell array,ÿ��cell�洢�ַ�������ģ�����ͣ���'[convolutional]'
moduleInfoList = []; % cell array,ÿ��cell�洢�ṹͼ��ģ����Ϣ
layerToModuleIndex = []; % ������n*1��vector,ÿ��ֵ����matlab�д�layersӳ�䵽module�����

%% 1������net��ģ��
module_idx = 1;
layerNames = [];% �ַ�����n*1��vector,ÿ��ֵ�洢ÿ��������֣�����shortcut,routeҪ�õ�
numsLayers = length(net.Layers);
for i = 1:numsLayers
    is_new_module = true;st = struct();
    layerNames = [layerNames;{net.Layers(i).Name}];
    currentLayerType = class(net.Layers(i));
    if strcmpi(currentLayerType,'nnet.cnn.layer.ImageInputLayer')
        moduleTypeList = [moduleTypeList;{'[net]'}];
        st = hyperParams;
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.Convolution2DLayer')
        moduleTypeList = [moduleTypeList;{'[convolutional]'}];
        layer = net.Layers(i);
        st = struct('filters',sum(layer.NumFilters),...
            'size',layer.FilterSize(1),...
            'pad',layer.PaddingSize(1),...
            'stride',layer.Stride(1),...
            'activation','linear');
    elseif strcmpi(currentLayerType, 'nnet.cnn.layer.GroupedConvolution2DLayer')
        moduleTypeList = [moduleTypeList;{'[convolutional]'}];
        layer = net.Layers(i);
        st = struct('groups',layer.NumGroups,...
            'filters',layer.NumGroups*layer.NumFiltersPerGroup,...
            'size',layer.FilterSize(1),...
            'pad',layer.PaddingSize(1),...
            'stride',layer.Stride(1),...
            'activation','linear');
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.FullyConnectedLayer')
        moduleTypeList = [moduleTypeList;{'[connected]'}];
        layer = net.Layers(i);
        st = struct('output',layer.OutputSize,...
            'activation','linear');
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.BatchNormalizationLayer')
        module_idx = module_idx-1;
        moduleInfoList{end}.batch_normalize = 1;
        is_new_module = false;
    elseif  strcmpi(currentLayerType,'nnet.cnn.layer.ReLULayer')
        module_idx = module_idx-1;
        moduleInfoList{end}.activation = 'relu';
        is_new_module = false;
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.LeakyReLULayer')
        module_idx = module_idx-1;
        moduleInfoList{end}.activation = 'leaky';
        is_new_module = false;
    elseif strcmpi(currentLayerType,'nnet.onnx.layer.ClipLayer')%������ֵΪ6����
        module_idx = module_idx-1;
        moduleInfoList{end}.activation = 'relu6'; %ʵ����������matlab��clippedReluLayer,6
        is_new_module = false;
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.ClippedReLULayer') %������ֵΪ6����
        module_idx = module_idx-1;
        moduleInfoList{end}.activation = 'relu6'; %ʵ����������matlab��clippedReluLayer,6
        is_new_module = false;
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.MaxPooling2DLayer')
        moduleTypeList = [moduleTypeList;{'[maxpool]'}];
        layer = net.Layers(i);
        if i==numsLayers-3||i==numsLayers-2 % ���һ�㣬�����Զ��ƶ�����ͼ��С
            st = struct();
        else
            if strcmp(layer.PaddingMode,'manual')
                st = struct('size',layer.PoolSize(1),...
                    'stride',layer.Stride(1),...
                    'padding',layer.PaddingSize(1));
            else
                st = struct('size',layer.PoolSize(1),...
                    'stride',layer.Stride(1));
            end
        end
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.AveragePooling2DLayer')
        moduleTypeList = [moduleTypeList;{'[avgpool]'}];
        layer = net.Layers(i);
        if i==numsLayers-3||i==numsLayers-2% ���һ�㣬�����Զ��ƶ�����ͼ��С
            st = struct();
        else
            if strcmp(layer.PaddingMode,'manual')
                st = struct('size',layer.PoolSize(1),...
                    'stride',layer.Stride(1),...
                    'padding',layer.PaddingSize(1));
            else
                st = struct('size',layer.PoolSize(1),...
                    'stride',layer.Stride(1));
            end
        end
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.SoftmaxLayer')
        moduleTypeList = [moduleTypeList;{'[softmax]'}];
        st = struct('groups',1);
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.AdditionLayer')
        moduleTypeList = [moduleTypeList;{'[shortcut]'}];
        st = struct('from',[],'activation','linear');
        layer_name = layerNames{i};
        index_Dlogical = startsWith(net.Connections.Destination,[layer_name,'/']);
        source = net.Connections.Source(index_Dlogical);
        index_Slogical = contains(layerNames(1:end-1),source);
        st.from = layerToModuleIndex(index_Slogical)-2; % -2 darknet module ���Ե�һ����[net]��ʼ��moduleΪ0�ļ���
        st.from = num2str(min(st.from)); % 2019.8.29�޸�
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.DepthConcatenationLayer')
        moduleTypeList = [moduleTypeList;{'[route]'}];
        st = struct('layers',[]);
        layer_name = layerNames{i};
        index_Dlogical = startsWith(net.Connections.Destination,[layer_name,'/']);
        source = net.Connections.Source(index_Dlogical);
        index_Slogical = contains(layerNames(1:end-1),source);
        st.layers = layerToModuleIndex(index_Slogical)-2; % -2 darknet module ���Ե�һ����[net]��ʼ��moduleΪ0�ļ���
        st.layers = join(string(st.layers),',');
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.DropoutLayer')
        moduleTypeList = [moduleTypeList;{'[dropout]'}];
        layer = net.Layers(i);
        st = struct('probability',layer.Probability);
    elseif strcmpi(currentLayerType, 'nnet.cnn.layer.ClassificationOutputLayer')
        continue;
    else
        moduleTypeList = [moduleTypeList;{'[unknow]'}];% ������Ҫ�ֶ���cfg�ļ����޸�
        st = struct('error',['unsupported this type:',currentLayerType,...
            ',you should manully modify it!']);
    end
    % ����
    if is_new_module
        moduleInfoList = [moduleInfoList;{st}];
    end
    layerToModuleIndex = [layerToModuleIndex;module_idx];
    module_idx = module_idx+1;
end % ��ֹ����

%% cutoff
if ~isempty(varargin)
    cutoffModule = varargin{1};
    moduleTypeList(cutoffModule+2:end) = [];
    moduleInfoList(cutoffModule+2:end) = [];
end

%% 2��д��cfgģ�������ļ�
assert(length(moduleTypeList)==length(moduleInfoList));
nums_module = length(moduleTypeList);
fid_cfg = fopen(cfgfileName,'w');
for i = 1:nums_module
    currentModuleType = moduleTypeList{i};% currentModuleType���ַ���������
    currentModuleInfo = moduleInfoList{i}; % currentModuleInfo��struct����
    % ���module����д��
    if i==1
        fprintf(fid_cfg,'%s\n','# This file is generated by MATLAB');% ע�Ͳ���
        fprintf(fid_cfg,'%s\n',currentModuleType);% module������
    else
        fprintf(fid_cfg,'%s\n',['# darknet module ID:',num2str(i-2)]); %cfg����ʽ����
        fprintf(fid_cfg,'%s\n',currentModuleType);% module������
    end
    
    fields = fieldnames(currentModuleInfo);
    for j = 1:length(fields) %д��module�Ľṹ����Ϣ
        fieldname = fields{j};
        fieldvalue = currentModuleInfo.(fieldname);
        fprintf(fid_cfg,'%s=%s\n',fieldname,num2str(fieldvalue));% module������
    end
    fprintf(fid_cfg,'\n');
end
fclose(fid_cfg);

%% 3������weightsȨ��
fid_weight = fopen(weightfileName,'wb');
fwrite(fid_weight,[0,2,5],'int32');% version
fwrite(fid_weight,0,'int64'); % number images in train
nums_module = length(moduleTypeList);
for module_index = 1:nums_module
    currentModuleType = moduleTypeList{module_index};% �ַ�����
    currentModuleInfo = moduleInfoList{module_index}; % struct
    currentModule = net.Layers(module_index == layerToModuleIndex);
    if strcmp(currentModuleType,'[convolutional]')||strcmp(currentModuleType,'[connected]')
        conv_layer = currentModule(1);
        % �����module��BN�����ȴ洢BN�Ĳ���
        if isfield(currentModuleInfo,'batch_normalize') % darknetһ���׶ˣ�������conv bias�Ĳ���
            bn_layer = currentModule(2);
            bn_bias = bn_layer.Offset;
            fwrite(fid_weight,bn_bias(:),'single');
            bn_weights = bn_layer.Scale;
            fwrite(fid_weight,bn_weights(:),'single');
            bn_mean = bn_layer.TrainedMean;
            fwrite(fid_weight,bn_mean(:),'single');
            bn_var = bn_layer.TrainedVariance;
            fwrite(fid_weight,bn_var(:),'single');
        else
            % conv bias
            conv_bias = conv_layer.Bias;
            conv_bias = permute(conv_bias,[2,1,3,4]);% ֧�� groupedConvolution2dLayer
            fwrite(fid_weight,conv_bias(:),'single');
        end
        % conv weights
        conv_weights = conv_layer.Weights;
        conv_weights = permute(conv_weights,[2,1,3,4,5]);% ֧�� groupedConvolution2dLayer
        fwrite(fid_weight,conv_weights(:),'single');
    end
end
fclose(fid_weight);

