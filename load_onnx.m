function out = load_onnx(nn_onnx)
    % We want to automate a parser to load onnx into NNV using MATLAB
    % Mathworks seems to have improved the development of the ONNX importer
    % If we can avoid writing a parser outside of MATLAB, it'd be easier
    % and better, so here it is

    % There are a few options that we can play with, so in the future we are going to
    % write a few nested try-catch statements and evauate if we can load
    % the networks without the inclusion of any custom layers

    % In this function we are going to test many ways we can attempt to load onnx networks, 
    % and create a log of succesful tries for each function. 

    % The VNN benchmarks are mostly CNNs, and we may need to define
    % different options in the input layer when loading CNNs vs FFNNs
    % (different input size, different output tasks...)

    %% Load network with no assumptions
    try
        net = importONNXNetwork(nn_onnx,'GenerateCustomLayers',false);
%         out.noCustom.success = true;
%         out.noCustom.results = net;
        out.noCustom = true;
    catch ME
%         out.noCustom.success = false;
%         out.noCustom.results = ME;
        out.noCustom = ME;
    end

    %% Regression - Assume the network is regression based with a feature layer input
    try 
        net = importONNXNetwork(nn_onnx, OutputLayerType="regression", GenerateCustomLayers = false);
        out.regression = true;
%         out.regression.result = net;
%         out.regression.success = true;
    catch ME
        out.regression = ME;
%         out.regression.result = ME;
%         out.regression.success = false;
    end

    try 
        net = importONNXNetwork(nn_onnx, OutputLayerType="regression", TargetNetwork="dlnetwork", GenerateCustomLayers = false);
        out.regression_dlnet = true;
%         out.regression_dlnet.result = net;
%         out.regression_dlnet.success = true;
    catch ME
        out.regression_dlnet = ME;
%         out.regression_dlnet.result = ME;
%         out.regression_dlnet.success = false;
    end

    try 
        net = importONNXNetwork(nn_onnx, InputDataFormats='BC' ,OutputLayerType="regression", TargetNetwork="dlnetwork", GenerateCustomLayers = false);
        out.regression_dlnet_bc = true;
%         out.regression_dlnet_bc.result = net;
%         out.regression_dlnet_bc.success = true;
    catch ME
        out.regression_dlnet_bc = ME;
%         out.regression_dlnet_bc.result = ME;
%         out.regression_dlnet_bc.success = false;
    end

    try 
        net = importONNXNetwork(nn_onnx, InputDataFormats='BSSC' ,OutputLayerType="regression", TargetNetwork="dagnetwork", GenerateCustomLayers = false);
        out.regression_dagnet = true;
%         out.regression_dagnet.result = net;
%         out.regression_dagnet.success = true;
    catch ME
        out.regression_dagnet = ME;
%         out.regression_dagnet.result = ME;
%         out.regression_dagnet.success = false;
    end

    %% Classification - Assume network is classification based, with image input layer
    try 
        net = importONNXNetwork(nn_onnx, InputDataFormats='BSSC' ,OutputLayerType="classification", TargetNetwork="dagnetwork", GenerateCustomLayers = false);
        out.class_dagnet = true;
%         out.class_dagnet.result = net;
%         out.class_dagnet.success = true;
    catch ME
        out.class_dagnet = ME;
%         out.class_dagnet.result = ME;
%         out.class_dagnet.success = false;
    end

    try 
        net = importONNXNetwork(nn_onnx, InputDataFormats='BSSC' ,OutputDataFormats='BSSC' ,OutputLayerType="classification", TargetNetwork="dagnetwork", GenerateCustomLayers = false);
        out.class_dagnet_out = true;
%         out.class_dagnet_out.result = net;
%         out.class_dagnet_out.success = true;
    catch ME
        out.class_dagnet_out = ME;
%         out.class_dagnet_out.result = ME;
%         out.class_dagnet_out.success = false;
    end

    try 
        net = importONNXNetwork(nn_onnx, InputDataFormats='BSSC' ,OutputLayerType="classification", TargetNetwork="dlnetwork", GenerateCustomLayers = false);
        out.class_dlnet = true;
%         out.class_dlnet.result = net;
%         out.class_dlnet.success = true;
    catch ME
        out.class_dlnet = ME;
%         out.class_dlnet.result = ME;
%         out.class_dlnet.success = false;
    end

    try 
        net = importONNXNetwork(nn_onnx, InputDataFormats='BSSC' ,OutputDataFormats='BSSC' ,OutputLayerType="classification", TargetNetwork="dlnetwork", GenerateCustomLayers = false);
        out.class_dlnet_out = true;
%         out.class_dlnet_out.result = net;
%         out.class_dlnet_out.success = true;
    catch ME
        out.class_dlnet_out = ME;
%         out.class_dlnet_out.result = ME;
%         out.class_dlnet_out.success = false;
    end
    
    %% Last tries 

    try 
        net = importONNXNetwork(nn_onnx, FoldConstants="deep",TargetNetwork="dagnetwork", GenerateCustomLayers=false);
        out.last_dagnet = true;
%         out.last_dagnet.sucess = true;
%         out.last_dagnet.results= net;
    catch ME
        out.last_dagnet = ME;
%         out.last_dagnet.sucess = false;
%         out.last_dagnet.result = ME;
    end

    try 
        net = importONNXNetwork(nn_onnx, FoldConstants="deep",TargetNetwork="dlnetwork", GenerateCustomLayers=false);
        out.last_noCustom = true;
%         out.last_noCustom.sucess = true;
%         out.last_noCustom.results= net;
    catch ME
        out.last_noCustom = ME;
%         out.last_noCustom.sucess = false;
%         out.last_noCustom.result = ME;
    end

    try 
        net = importONNXNetwork(nn_onnx, FoldConstants="deep");
        out.last = true;
%         out.last.sucess = true;
%         out.last.result = net;
    catch ME
        out.last = ME;
%         out.last.sucess = false;
%         out.last.result = ME;
    end

     %% Last try optimizing loading of layers (may reduce # of custom layers)

    try 
        net = importONNXLayers(nn_onnx, FoldConstants="deep");
        out.layers = true;
%         out.layers.sucess = true;
%         out.layers.result = net;
    catch ME
        out.layers = ME;
%         out.layers.sucess = false;
%         out.layers.result = ME;
    end

end

