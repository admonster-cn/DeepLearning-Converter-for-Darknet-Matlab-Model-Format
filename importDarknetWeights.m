function [lgraph,hyperParams,numsNetParams,FLOPs] = importDarknetWeights(cfgfile,weightsfile,cutoffModule)
% IMPORTDARKNETWEIGHTS ���ܣ�ָ�����벿��module��darknetģ��
% ���룺cfgfile, ָ����cfg��׺��ģ�������ļ���Ŀǰֻ֧��series network����DAGnetwork
%       weighfile, ָ����.weights��׺�Ķ������ļ���Ŀǰֻ֧��series network����DAGnetwork
%       cutoffModule,(��ѡ��)1*1����������ָ������darknetǰcutoffModule��module����1Ϊbase��������û�и�����������������
% �����lgraph�� matlab���ѧϰģ�ͼ���ͼ��Ŀǰֻ֧��series network����DAGnetwork
%      hyperParams,�ṹ�壬���������ļ�
%      numsReadParams,Ȩ�ز�������
%      FLOPs�� ģ�ͼ�����
% ע�⣺1���ʺ�2019a�汾������
%       2��leaky��ֵĿǰȡ��0.1
%       3�����ĳ��module����bn�㣬��conv��biasΪ0����Ϊdarknet�����ִ洢��ʽ
%      4�������뵽yolo�㣬�˳����룬��ʱ��֧��yolo������Ĳ㣬��Ϊyolov3�ٷ�����֧��
%      5��darknet weights����˳������ΪBN��offset,scale,mean,variance,Conv���bias,weights
%      ����ͼ���output Size = (Input Size �C ((Filter Size �C 1)*Dilation Factor + 1) + 2*Padding)/Stride + 1
% �ο���1���ٷ��ĵ���Specify Layers of Convolutional Neural Network
%      2��https://www.zhihu.com/question/65305385
%       3��https://github.com/ultralytics/yolov3/blob/master/models.py
% cuixingxing150@gmail.com
% 2019.8.19
%

[lgraph,hyperParams,numsNetParams,FLOPs,...
    moduleTypeList,moduleInfoList,layerToModuleIndex] = importDarkNetLayers(cfgfile,cutoffModule);
assert(length(moduleTypeList)==length(moduleInfoList));

%% ��ȡȨ�ز����ļ�
fid_w = fopen(weightsfile,'rb');
header = fread(fid_w, 3, '*int32');
if header(2) > 1
    header2 = fread(fid_w, 1, '*int64'); % int64ռ��8���ֽ�
else
    header2 = fread(fid_w, 1, '*int32'); % int32ռ��4���ֽ�
end
fprintf('Major :%d, Minor :%d,Revision :%d,number of images during training:%d,reading params...\n',...
    header(1),header(2),header(3),header2);
weights = fread(fid_w,'*single');
fclose(fid_w);

% numsWeightsParams = numel(weights);
readSize = 1;
numsModule = length(moduleTypeList);

for i = 1:numsModule
    currentModuleType = moduleTypeList{i};
    currentModuleInfo = moduleInfoList{i};
    if strcmp(currentModuleType,'[convolutional]')
        currentModule = lgraph.Layers(i==layerToModuleIndex);
        filterSize = str2double(currentModuleInfo.size);
        numFilters = str2double(currentModuleInfo.filters);
        channels_in = moduleInfoList{i-1}.channels;
            
        if isfield(currentModuleInfo,'batch_normalize')
            % bn bias
            bn_bias = weights(readSize:readSize+numFilters-1);
            bn_bias = reshape(bn_bias,[1,1,numFilters]);
            currentModule(2).Offset = bn_bias;
            readSize = readSize+numFilters;
            % bn weight
            bn_weight = weights(readSize:readSize+numFilters-1);
            bn_weight = reshape(bn_weight,[1,1,numFilters]);
            currentModule(2).Scale = bn_weight;
            readSize = readSize+numFilters;
             % bn trainedMean
            bn_mean = weights(readSize:readSize+numFilters-1);
            bn_mean = reshape(bn_mean,[1,1,numFilters]);
            currentModule(2).TrainedMean = bn_mean;
            readSize = readSize+numFilters;
             % bn trainedVariance
            bn_var = weights(readSize:readSize+numFilters-1);
            bn_var = reshape(bn_var,[1,1,numFilters]);
            if any(bn_var<-0.01)
                error("����Ӧ�ô���0��");
            end
            currentModule(2).TrainedVariance = abs(bn_var); % ��ֹ�ӽ���0�����Ǹ���
            readSize = readSize+numFilters;
            % conv bias Ϊ0
            if isfield(currentModuleInfo,'groups')
                numGroups = str2double(currentModuleInfo.groups);
                numFiltersPerGroup_out = numFilters/numGroups;
                currentModule(1).Bias = zeros(1,1,numFiltersPerGroup_out,numGroups,'single');
            else
                currentModule(1).Bias = zeros(1,1,numFilters,'single');
            end
        else
            % load conv bias
            conv_bias = weights(readSize:readSize+numFilters-1);
            if isfield(currentModuleInfo,'groups')
                numGroups = str2double(currentModuleInfo.groups);
                numFiltersPerGroup_out = numFilters/numGroups;
                conv_bias = reshape(conv_bias,1,1,numFiltersPerGroup_out,numGroups);
            else
                conv_bias = reshape(conv_bias,1,1,numFilters);
            end
            currentModule(1).Bias = conv_bias;
            readSize = readSize+numFilters;
        end % end of is bn
        % load conv weights
        if isfield(currentModuleInfo,'groups')
            numGroups = str2double(currentModuleInfo.groups);
            numFiltersPerGroup_out = numFilters/numGroups;
            nums_conv_w = filterSize*filterSize*channels_in/numGroups*numFiltersPerGroup_out*numGroups;
            conv_weights = weights(readSize:readSize+nums_conv_w-1);
            conv_weights = reshape(conv_weights,filterSize,filterSize,channels_in/numGroups,numFiltersPerGroup_out,numGroups);
            conv_weights = permute(conv_weights,[2,1,3,4,5]);
            currentModule(1).Weights = conv_weights;
            readSize = readSize+nums_conv_w;
        else
            nums_conv_w = filterSize*filterSize*channels_in*numFilters;% weights
            conv_weights = weights(readSize:readSize+nums_conv_w-1);
            conv_weights = reshape(conv_weights,filterSize,filterSize,channels_in,numFilters);
            conv_weights = permute(conv_weights,[2,1,3,4]);
            currentModule(1).Weights = conv_weights;
            readSize = readSize+nums_conv_w;
        end % end of load conv weights  
        % ���²���
        % lgraph.Layers(i==layerToModuleIndex) = currentModule;
        for replaceInd = 1:length(currentModule)
            layerName = currentModule(replaceInd).Name;
            lgraph = replaceLayer(lgraph,layerName,currentModule(replaceInd));
        end
    end % end of module '[convolutional]'
    
    % fullyConnectedLayer 
    if  strcmp(currentModuleType,'[connected]')
        currentModule = lgraph.Layers(i==layerToModuleIndex);
        numFilters = str2double(currentModuleInfo.output);
        % load fc bias
        numBias = numFilters;
        fl_bias = weights(readSize:readSize+numBias-1);
        fl_bias = reshape(fl_bias,numBias,1);
        currentModule(1).Bias = fl_bias;
        readSize = readSize+numBias;
         % load fc weights
        input_all_neurons = prod(moduleInfoList{i-1}.mapSize)*moduleInfoList{i-1}.channels;
        numWeights = numFilters*input_all_neurons; % help fullyConnectedLayer weights
        fl_weights = weights(readSize:readSize+numWeights-1);
        fl_weights = reshape(fl_weights,input_all_neurons,numFilters);
        fl_weights = permute(fl_weights,[2,1]);% fc����Ҫpermute?
        currentModule(1).Weights = fl_weights;
        readSize = readSize+numWeights;
        % ���²���
        for replaceInd = 1:length(currentModule)
            layerName = currentModule(replaceInd).Name;
            lgraph = replaceLayer(lgraph,layerName,currentModule(replaceInd));
        end
    end % end of module '[connected]'
end % end of nums of module

if isa(lgraph.Layers(end),'nnet.cnn.layer.SoftmaxLayer')
    lastLayerName = lgraph.Layers(end).Name;
    classifyLayer = classificationLayer('Name','classify');
    lgraph = addLayers(lgraph,classifyLayer);
    lgraph = connectLayers(lgraph,lastLayerName,'classify');
end

fprintf('Load parameters succfully!\n')

