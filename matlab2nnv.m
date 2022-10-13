function net = matlab2nnv(Mnetwork)

%% Check for correct inputs and process network
ntype = class(Mnetwork); % input type
if ~contains(["SeriesNetwork", "LayerGraph", "DAGNetwork", "dlnetwork"], ntype)
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
    error('Sorry, we currently do not support this type of neural networks. \n Netowkrs must be fullycoonected, we do not support skipped or sparse connections');
end

%% Transform to NNV (to implement)
% For now, just check if we could support it
supportLayers = ["nnet.cnn.layer.ImageInputLayer"; % List of supported layers (parse in  CNN)
    "nnet.cnn.layer.Convolution2DLayer";
    "nnet.cnn.layer.ReLULayer";
    "nnet.cnn.layer.BatchNormalizationLayer";
    "nnet.cnn.layer.MaxPooling2DLayer";
    "nnet.cnn.layer.AveragePooling2DLayer";
    "nnet.cnn.layer.FullyConnectedLayer";
    "nnet.cnn.layer.PixelClassificationLayer";
    "nnet.keras.layer.FlattenCStyleLayer";
    "nnet.cnn.layer.FlattenLayer";
    "nnet.onnx.layer.FlattenLayer";
    "nnet.onnx.layer.SigmoidLayer";
    "nnet.onnx.layer.ElementwiseAffineLayer"];

n = length(Layers);
nnvLayers = [ ];

for i=1:n
    L = Layers(i);
    if isa(L, 'nnet.cnn.layer.DropoutLayer') || isa(L, 'nnet.cnn.layer.SoftmaxLayer') || isa(L, 'nnet.cnn.layer.ClassificationOutputLayer') || isa(L,"nnet.onnx.layer.VerifyBatchSizeLayer") ...
            || isa(L, "nnet.onnx.layer.RegressionOutoutLayer")
        fprintf('\nLayer %d is a %s class which is neglected in the analysis phase', i, class(L));                   
        if isa(L, 'nnet.cnn.layer.ClassificationOutputLayer')
            outputSize = L.OutputSize;
        end
    else                    
        if contains(supportLayers, class(L))
            fprint('\n Including layer %d, a %s, into the NNV analysis', i, class(L))
            NNVLayers = [nnvLayers, L];
        else
%             if contains(class(L),'Flatten') && isempty(nnvLayers)
%                 fprintf('\nLayer %d is a %s . For this analysis, we can discard this function for now', i, class(L));
            fprintf('\nLayer %d is a %s which have not supported yet in nnv, please consider removing this layer for the analysis', i, class(L));
            error('\nUnsupported Class of Layer');                     
        end
    end
end

%% Future todos
% Check the layer and parse each layer how we have it in NNV
% Add a variable tp check which type of network we have (semantic
% segmentation, CNN, feedforward...). This could help dealing with some of
% the custom flatten layers. Remove that, "squeeze" input size and reshape
% based on parameters of next layer.

end

