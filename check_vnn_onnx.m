%
% probinette
%
% Checks each VNN onnx model to see if it can be loaded into Matlab
%
function f = check_vnn_onnx(folder)

path = strcat('/Users/probinet/SANDBOX/vnncomp2022_benchmarks/benchmarks/', folder, '/onnx/');

listing = dir(path);
onnx_nets = {listing.name};
isGitKeep = contains(onnx_nets, '.gitkeep');
onnx_nets([listing.isdir]|isGitKeep) = [];

layer_fail_count = 0;
network_fail_count = 0;
networks_failed = {};
layers_failed = {};

% for each file path attempt both importONNXLayer and importONNXNetwork
for net = onnx_nets
    X = ['Attempting: ', net{1}];
    disp(X);

    % attempt layer import
    try
        importONNXLayers(net{1});
    catch
%         disp('Layers Unsuccessful')
        layer_fail_count = layer_fail_count + 1;
        layers_failed = [layers_failed; net{1}];
    end

    % attempt network import
    try
        importONNXNetwork(net{1});
    catch
%         disp('Network Unsuccessful')
        network_fail_count = network_fail_count + 1;
        networks_failed = [networks_failed; net{1}];
    end
end

disp(folder);
disp("----- Import Layer Stats with importONNXLayers ------ ");
disp("Count Failed: ");
disp(layer_fail_count);
disp("Nets Failed: ");
disp(layers_failed);
disp("----- Import Network Stats with importONNXNetwork ------ ");
disp("Count Failed: ");
disp(network_fail_count);
disp("Nets Failed: ");
disp(networks_failed);
