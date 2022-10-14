%% This is a script with a set of examples from the VNN competition
% We are going to load several neural networks using the function onnx2nnv

% vnn comp folder
vnnFolder = "/home/manzand/Documents/MATLAB/vnncomp2022_benchmarks/benchmarks/"; 
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
cifarSmallFile = "cifar100_tinyimagenet_resnet/onnx/CIFAR100_resnet_small.onnx";
cifarSmall = onnx2nnv(vnnFolder + cifarSmallFile);


