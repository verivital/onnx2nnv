%% This is a script with a set of examples from the VNN competition
% We are going to load several neural networks using the function onnx2nnv

% vnn comp folder
% vnnFolder = "/home/manzand/Documents/MATLAB/vnncomp2022_benchmarks/benchmarks/";
vnnFolder = "/home/dieman95/Documents/MATLAB/vnncomp2022_benchmarks/benchmarks/";
% To replicate the results, plesae update the vnn comp folder as well as
% individual files (onnx) for each benchmark


%% Example 1 -- ACAS Xu neural networks
% Define file to load and loading options
acasFile = "acasxu/onnx/ACASXU_run2a_2_4_batch_2000.onnx";
acasOptions.InputDataFormat = 'BCSS'; % loading options
acas1 = onnx2nnv(vnnFolder + acasFile, acasOptions);

acasFile = "acasxu/onnx/ACASXU_run2a_2_4_batch_2000.onnx";
% acasOptions.InputDataFormat = 'BCSS'; % loading options
acas2 = onnx2nnv(vnnFolder + acasFile);

%% Example 2 -- carvana unets (not supported)
% No need to run

% carvanaSimpFile = "carvana_unet_2022/onnx/unet_simp_small.onnx";
% try
%     carvanaSimp = onnx2nnv(vnnFolder + carvanaSimpFile);
% catch 
%     warning('Unets are not yet suported');
% end

%% Example 3 -- CIFAR100 tinyImagenet Resnet (not supported)
% Resnets are not yet supported, although all the layers in it are
% Working on adding support

% subFolder = 'cifar100_tinyimagenet_resnet/onnx';
try
    cifarSmallFile = "cifar100_tinyimagenet_resnet/onnx/CIFAR100_resnet_small.onnx";
    cifarSmall = onnx2nnv(vnnFolder + cifarSmallFile);
catch ME_cifarSmall
    warning('Resnets are not yet supported')
end

%% Example 4 -- CIFAR2020
% cifar2020File = "cifar2020/onnx/cifar10_2_255.onnx";
% cifar2020_options = struct;
% cifar2020_options.OutputDataFormat = "BC";
% cifar2020 = onnx2nnv(vnnFolder+cifar2020File, cifar2020_options);

% This generates a matlab network with a shape to reshape layer (unsupported)
% This layer has ONNXParameters.NonLearnables.UnsqueezeAxesXXXX
% Need to double check, but the numbers after that variable might be the
% order in which the dimensions of the input to this layer "unsqueeze" to
% form a vector for the next layer

cifarSimplifiedFile = "cifar2020/onnx/cifar10_2_255_simplified.onnx";
cifarSimplified = onnx2nnv(vnnFolder + cifarSimplifiedFile);

convReluFile = "cifar2020/onnx/convBigRELU__PGD.onnx";
convRelu = onnx2nnv(vnnFolder + convReluFile);

%% Example 5 -- cifarbiasfield
cifarBias0File = "cifar_biasfield/onnx/cifar_bias_field_0.onnx";
% cifarBias0 = onnx2nnv(vnnFolder+cifarBias0File);
% net = importONNXLayers(vnnFolder+cifarBias0File, InputDataFormats="BC");

%% Example 6 -- colins_rul_cnn
rulFull20File = "collins_rul_cnn/onnx/NN_rul_full_window_20.onnx";
rulFull20 = onnx2nnv(vnnFolder+rulFull20File);

% This example looks like it has a conv2D layer that should be a fc layer
% Need to add support for these type of occasions
% Either convert it to a fc layer (check that filter sizes and all of that
% match the fc), or add support within the conv2D layer. I think
% trasformation will be easier
