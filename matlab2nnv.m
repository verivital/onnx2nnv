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
% TODO: 
% add support to resnets and unets, at least the ones created in MATLAB

%% Transform to NNV 

n = length(Layers);
nnvLayers = {};
count = 1; % number of layers added to NNV
names = [];
indxs = [];
replace_layers = {};
countRep = 1;

for i=1:n
    L = Layers(i);
    if isa(L, 'nnet.cnn.layer.DropoutLayer') || isa(L, 'nnet.cnn.layer.SoftmaxLayer') || isa(L, 'nnet.cnn.layer.ClassificationOutputLayer') ...
            || isa(L,"nnet.onnx.layer.VerifyBatchSizeLayer") || isa(L, "nnet.cnn.layer.RegressionOutputLayer")
        fprintf('Layer %d is a %s class which is neglected in the analysis phase \n', i, class(L));
        % Substitute their names in destinations and sources
        % Or even easier, assign this name to match the previous layer, easier to refactor
        if isa(L, 'nnet.cnn.layer.ClassificationOutputLayer')
            outputSize = L.OutputSize;
        end
        if exist('Li','var')
            replace_layers{countRep} = {Li.Name; L.Name};
            countRep = countRep + 1;
        end
        continue;
    else
        fprintf('\nParsing Layer %d... \n', i);
        if isa(L, 'nnet.cnn.layer.ImageInputLayer')
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
            if ~isempty(nnvLayers) && isa(nnvLayers{end}, 'ImageInputLayer')
                nnvLayers{end} = []; % remove image input layer is followed by fullyconnected layer, input is set by dimensions of weights
                count = count -1;
            end
        elseif isa(L, 'nnet.cnn.layer.PixelClassificationLayer')
            Li = PixelClassificationLayer.parse(L);
        elseif isa(L, 'nnet.keras.layer.FlattenCStyleLayer') || isa(L, 'nnet.cnn.layer.FlattenLayer') || isa(L, 'nnet.onnx.layer.FlattenLayer') ...
                || isa(L, 'nnet.onnx.layer.FlattenInto2dLayer')
            Li = FlattenLayer.parse(L);
        elseif isa(L, 'nnet.keras.layer.SigmoidLayer') || isa(L, 'nnet.onnx.layer.SigmoidLayer')
            Li = SigmoidLayer.parse(L);
        elseif isa(L, 'nnet.onnx.layer.ElementwiseAffineLayer')
            Li = ElementwiseAffineLayer.parse(L);
        elseif contains(class(L), ["flatten"; "Flatten"])
            if isempty(nnvLayers) || isa(nnvLayers{end}, 'ImageInputLayer')
                fprintf('Discard flatten layer for analysis if directly after the input. \n')
                continue
            else
                fprintf('Layer %d is a %s which have not supported yet in nnv, please consider removing this layer for the analysis \n', i, class(L));
                error('Unsupported Class of Layer');
            end
        elseif isa(L, 'nnet.cnn.layer.FeatureInputLayer')
            if isempty(L.Mean) && isempty(L.StandardDeviation) && isempty(L.Min) && isempty(L.Max)
                fprintf('Layer %d is a %s class which is neglected in the analysis phase \n', i, class(L));
                fprintf('No normalization or transformation is done on the input. \n')
                continue;
            else
                Li = FeatureInputLayer.parse(); % TODO: Need to create a FeatureInputLayer class and implement this
            end
        elseif contains(class(L, "ReshapeLayer"))
            Li = ReshapeLayer(L);
        else
            fprintf('Layer %d is a %s which have not supported yet in nnv, please consider removing this layer for the analysis \n', i, class(L));
            error('Unsupported Class of Layer');                     
        end
        % Add layer
        nnvLayers{count} = Li;
        names = [names; string(L.Name)];
        indxs = [indxs; count];
%         name2number(Li.Name,count);
        count = count + 1; 
    end
end
name2number = containers.Map(names,indxs);

% Next step: Connections
sources = conns.Source;
dests = conns.Destination;

% Convert connection to number (corresponding index in array)
for k= 1:length(sources)
    if any(contains(names, sources{k}))
        sources{k} = name2number(sources{k});
    end
    if any(contains(names, dests{k}))
        dests{k} = name2number(dests{k});
    end
end

% Change all other connections to previous layer
for k = 1:length(sources)
    if ~isnumeric(dests{k})
        dests{k} = sources{k};
    end
    if ~isnumeric(sources{k})
        if k > 1
            sources{k} = dests{k-1};
        else
            sources{k} = 1;
        end
    end
end

% Remove all duplicate connections
new_sources = [];
new_dests = [];
for k=1:length(sources)
    if sources{k} ~= dests{k}
        new_sources = [new_sources; sources{k}];
        new_dests = [new_dests; dests{k}];
    else
        if k == length(sources)
            new_dests = [new_dests; dests{k}+1];
            new_sources = [new_sources; sources{k}];
        end
    end
end

ConnectionsTable = table(new_sources, new_dests, 'VariableNames', {'Source', 'Destination'});

% Create neural network
net = NN(nnvLayers, ConnectionsTable);

%% Future todos
% Check the layer and parse each layer how we have it in NNV
% Add a variable tp check which type of network we have (semantic
% segmentation, CNN, feedforward...). This could help dealing with some of
% the custom flatten layers. Remove that, "squeeze" input size and reshape
% based on parameters of next layer.

% function layer = parse_elementWise(L, Lprev)
%     layer = [];
%     if ~L.DoScale && ~L.DoOffset
%         fprintf('We can neglect this Elementwise Affine Layer, no operations added. \n')
%     elseif ~L.DoScale && L.DoOffset
%         if all(L.Offset == 0)
%             fprintf('We can neglect this Elementwise AffineLayer, no operations added. \n')
%         else
%             if isa(Lprev, 'FullyConnectedLayer')
%                 if all(Lprev.Bias == 0)
%                     b = reshape(L.Offset, [], 1); % ensure bias is a column vector
%                     Lprev.Bias = b; % update last layer, reduce number of layers
%                     layer = {Lprev}; % change it to a cell to substitute the previous layer
%                     fprintf('Adding weights to the previous layer. \n')
%                 else
%                     layer = FullyConnectedLayer(L.Scale, L.Offset);
%                     fprintf('Creating a fullyconnected layer with only bias vector in place of the Elementwise Affina Layer. \n')
%                 end
%             else
%                 layer = FullyConnectedLayer([],reshape(L.Offset, [], 1)); % no weights, just bias
%             end
%         end
%     else
%         if all(L.Scale == 1) && all(L.Offset == 0)
%             fprintf('We can neglect this Elementwise AffineLayer, no operations added. \n')
%         else
%             b = reshape(L.Offset, [], 1); % ensure bias is a column vector
%             if size(L.Scale,1) == size(b,1)
%                 W = L.Scale;
%             else
%                 W = L.Scale';
%             end
%             layer = FullyConnectedLayer(W,b);
%             fprintf('Creating a fullyconnected layer in place of the Elementwise Affine Layer. \n')
%         end
%     end
% end

end

