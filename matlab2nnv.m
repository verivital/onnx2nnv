function net = matlab2nnv(Mnetwork)

%% Check for correct inputs and process network
ntype = class(Mnetwork); % input type
if ~contains(ntype, ["SeriesNetwork", "LayerGraph", "DAGNetwork", "dlnetwork"])
    error('Wrong input type. Input must be a SeriesNetwork, LayerGraph, DAGNetwork, or dlnetwork');
end

%% Process input types
% Input is a MATLAB type neural network (layergraph, seriesNetwork, dlnetwork or dagnetwork)
if ntype== "SeriesNetwork"
    conns = LayerGraph(Mnetwork).Connections; % get the table of connections
else
    conns = Mnetwork.Connections; % get the table of connections
end

Layers = Mnetwork.Layers; % get the list of layers

% Check if network is fullyconnected (no skip connections)
sources = unique(conns.Source);
targets = unique(conns.Destination);
% If a layer has multiple connections throw error. NNV does not support them yet.
if length(sources) ~= height(conns) || length(targets) ~= height(conns)
    error('Sorry, we currently do not support this type of neural networks. NNV does not support skipped or sparse connections');
end
% We need to update this in the future, but let's keep it simple for now
conns = 'default';

%% Transform to NNV (to implement)
% For now, just check if we could support it
% supportLayers = ["nnet.cnn.layer.ImageInputLayer"; % List of supported layers (parse in  CNN)
%     "nnet.cnn.layer.Convolution2DLayer";
%     "nnet.cnn.layer.ReLULayer";
%     "nnet.cnn.layer.BatchNormalizationLayer";
%     "nnet.cnn.layer.MaxPooling2DLayer";
%     "nnet.cnn.layer.AveragePooling2DLayer";
%     "nnet.cnn.layer.FullyConnectedLayer";
%     "nnet.cnn.layer.PixelClassificationLayer";
%     "nnet.keras.layer.FlattenCStyleLayer";
%     "nnet.cnn.layer.FlattenLayer";
%     "nnet.onnx.layer.FlattenLayer";
%     "nnet.onnx.layer.SigmoidLayer";
%     "nnet.onnx.layer.ElementwiseAffineLayer"];

n = length(Layers);
nnvLayers = {};
count = 1; % nuber of layers added to NNV

for i=1:n
    L = Layers(i);
    if isa(L, 'nnet.cnn.layer.DropoutLayer') || isa(L, 'nnet.cnn.layer.SoftmaxLayer') || isa(L, 'nnet.cnn.layer.ClassificationOutputLayer') ...
            || isa(L,"nnet.onnx.layer.VerifyBatchSizeLayer") || isa(L, "nnet.cnn.layer.RegressionOutputLayer")
        fprintf('Layer %d is a %s class which is neglected in the analysis phase \n', i, class(L));                   
        if isa(L, 'nnet.cnn.layer.ClassificationOutputLayer')
            outputSize = L.OutputSize;
        end
    else
        fprintf('\nParsing Layer %d... \n', i);
        if isa(L, "nnet.onnx.layer.ElementwiseAffineLayer")
            layer = parse_elementWise(L, nnvLayers{count-1});
            if ~isempty(layer)
                if isa(layer, 'cell')
                    Li = layer{1}; % Substitute layer (bias added)
                    count = count - 1;
%                     continue
                else
                    Li = layer; % add a layer (fullyconnected)
                end
            else
                continue
            end        
        elseif isa(L, 'nnet.cnn.layer.ImageInputLayer')
            Li = ImageInputLayer.parse(L);
        elseif isa(L, 'nnet.cnn.layer.Convolution2DLayer') 
            Li = Conv2DLayer.parse(L);
        elseif isa(L, 'nnet.cnn.layer.ReLULayer')
            Li = ReluLayer.parse(L);
        elseif isa(L, 'nnet.cnn.layer.BatchNormalizationLayer')
            Li = BatchNormalizationLayer.parse(L);
        elseif isa(L, 'nnet.cnn.layer.MaxPooling2DLayer')
            Li = MaxPooling2DLayer.parse(L);
        elseif isa(L, 'nnet.cnn.layer.AveragePooling2DLayer')
            Li = AveragePooling2DLayer.parse(L);
        elseif isa(L, 'nnet.cnn.layer.FullyConnectedLayer')
            Li = FullyConnectedLayer.parse(L);
            if isa(nnvLayers{end}, 'ImageInputLayer')
                nnvLayers{end} = []; % remove image input layer is followed by fullyconnected layer, input is set by dimensions of weights
                count = count -1;
            end
        elseif isa(L, 'nnet.cnn.layer.PixelClassificationLayer')
            Li = PixelClassificationLayer.parse(L);
        elseif isa(L, 'nnet.keras.layer.FlattenCStyleLayer') || isa(L, 'nnet.cnn.layer.FlattenLayer') || isa(L, 'nnet.onnx.layer.FlattenLayer')
            Li = FlattenLayer.parse(L);
        elseif isa(L, 'nnet.keras.layer.SigmoidLayer') || isa(L, 'nnet.onnx.layer.SigmoidLayer')
            Li = SigmoidLayer.parse(L);
%         elseif isa(L, 'nnet.onnx.layer.ElementwiseAffineLayer')
%             Li = ElementwiseAffineLayer.parse(L);
        elseif contains(class(L), ["flatten"; "Flatten"])
            if isempty(nnvLayers) || isa(nnvLayers{end}, 'ImageInputLayer')
                fprintf('Discard flatten layer for analysis if directly after the input. \n')
                continue
            else
                fprintf('Layer %d is a %s which have not supported yet in nnv, please consider removing this layer for the analysis \n', i, class(L));
                error('Unsupported Class of Layer');
            end
        else
%             if contains(class(L),'Flatten') && isempty(nnvLayers)
%                 fprintf('\nLayer %d is a %s . For this analysis, we can discard this function for now', i, class(L));
            fprintf('Layer %d is a %s which have not supported yet in nnv, please consider removing this layer for the analysis \n', i, class(L));
            error('Unsupported Class of Layer');                     
        end
        % Add layer
        nnvLayers{count} = Li;
        count = count + 1; 
    end
end

% Create neural network
net = NN(nnvLayers,conns);

%% Future todos
% Check the layer and parse each layer how we have it in NNV
% Add a variable tp check which type of network we have (semantic
% segmentation, CNN, feedforward...). This could help dealing with some of
% the custom flatten layers. Remove that, "squeeze" input size and reshape
% based on parameters of next layer.

function layer = parse_elementWise(L, Lprev)
    layer = [];
    if ~L.DoScale && ~L.DoOffset
        fprintf('We can neglect this Elementwise Affine Layer, no operations added. \n')
    elseif ~L.DoScale && L.DoOffset
        if all(L.Offset == 0)
            fprintf('We can neglect this Elementwise AffineLayer, no operations added. \n')
        else
            if isa(Lprev, 'FullyConnectedLayer')
                if all(Lprev.Bias == 0)
                    b = reshape(L.Offset, [], 1); % ensure bias is a column vector
                    Lprev.Bias = b; % update last layer, reduce number of layers
                    layer = {Lprev}; % change it to a cell to substitute the previous layer
                    fprintf('Adding weights to the previous layer. \n')
                else
                    layer = FullyConnectedLayer(L.Scale, L.Offset);
                    fprintf('Creating a fullyconnected layer with only bias vector in place of the Elementwise Affina Layer. \n')
                end
            end
        end
    else
        if all(L.Scale == 1) && all(L.Offset == 0)
            fprintf('We can neglect this Elementwise AffineLayer, no operations added. \n')
        else
            b = reshape(L.Offset, [], 1); % ensure bias is a column vector
            if size(L.Scale,1) == size(b,1)
                W = L.Scale;
            else
                W = L.Scale';
            end
            layer = FullyConnectedLayer(W,b);
            fprintf('Creating a fullyconnected layer in place of the Elementwise Affine Layer. \n')
        end
    end
end

end

