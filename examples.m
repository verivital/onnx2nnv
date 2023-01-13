%% This is a script with a set of examples from the VNN competition
% We are going to load several neural networks using the function onnx2nnv

% vnn comp folder
% vnnFolder = "/home/manzand/Documents/MATLAB/vnncomp2022_benchmarks/benchmarks/";
vnnFolder = "/home/dieman95/Documents/MATLAB/vnncomp2022_benchmarks/benchmarks/";
% To replicate the results, plesae update the vnn comp folder as well as
% individual files (onnx) for each benchmark

% Assign same reach options for all cases
reachOptions = struct;
reachOptions.reachMethod = 'approx-star';
reachOptions.dis_opt = 'display'; 

%% Example 1 -- ACAS Xu neural networks
% Define file to load and loading options
acasFile = "acasxu/onnx/ACASXU_run2a_2_4_batch_2000.onnx";
acasOptions.InputDataFormat = 'BCSS'; % loading options
acas1 = onnx2nnv(vnnFolder + acasFile, acasOptions);

% acasFile = "acasxu/onnx/ACASXU_run2a_2_4_batch_2000.onnx";
% acasOptions.InputDataFormat = 'BCSS'; % loading options
% acas2 = onnx2nnv(vnnFolder + acasFile);

% Evaluation example
x = [0;0;0;0;0];
y = acas1.evaluate(x);
% Reachability example
lb = [0;0;0;0;0];
ub = [0;0;0;0;0];
X = ImageStar(lb,ub);
Y = acas1.reach(X, reachOptions);

%% Example 2 -- carvana unets (not supported)
% No need to run (onnx support from matlan for unets is not great, leave this for the end)

carvanaSimpFile = "carvana_unet_2022/onnx/unet_simp_small.onnx";
try
    carvanaSimp = onnx2nnv(vnnFolder + carvanaSimpFile);
catch 
    warning('Unets are not yet suported');
end

%% Example 3 -- CIFAR100 tinyImagenet Resnet (not supported)
% Resnets are not yet supported, although all the layers in it are
% Working on adding support
% TODO: add addition / sum layer support and we can verify these

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
% form a vector for the next layer (Not true, seem to be randomly generated)

cifarSimplifiedFile = "cifar2020/onnx/cifar10_2_255_simplified.onnx";
cifarSimplified = onnx2nnv(vnnFolder + cifarSimplifiedFile);
lb = zeros([32 32 3]);
ub = zeros([32 32 3]);
X = ImageStar(lb,ub);
Y = cifarSimplified.reach(X, reachOptions);

convReluFile = "cifar2020/onnx/convBigRELU__PGD.onnx";
convReluOpts.InputDataFormat = 'BCSS';
convRelu = onnx2nnv(vnnFolder + convReluFile, convReluOpts);
lb = zeros([32 32 3]);
ub = zeros([32 32 3]);
X = ImageStar(lb,ub);
% Y = convRelu.reach(X, reachOptions); % try relax start, too slow
% otherwise (too many lp operation to solve in ReLU layers)

%% Example 5 -- cifarbiasfield
cifarBias0File = "cifar_biasfield/onnx/cifar_bias_field_0.onnx";
loadBias.InputDataFormat = "BC";
cifarBias0 = onnx2nnv(vnnFolder+cifarBias0File, loadBias);
% net = importONNXLayers(vnnFolder+cifarBias0File, InputDataFormats="BC");

%% Example 6 -- colins_rul_cnn
rulFull20File = "collins_rul_cnn/onnx/NN_rul_full_window_20.onnx";
rulFull20 = onnx2nnv(vnnFolder+rulFull20File);

% This example looks like it has a conv2D layer that should be a fc layer (could be both, fc and conv2d, easily interchangeable when parameters are right)
% Need to add support for these type of occasions (added support for
% special cases like when the number of filters (NumFilters = 1)
lb = zeros(20,20,1);
ub = zeros(20,20,1);
X = ImageStar(lb,ub);
Y_rul_a = rulFull20.reach(X, reachOptions); % Seems to be working
rulOptions = reachOptions;
rulOptions.reachMethod = 'exact-star';
Y_rul_e = rulFull20.reach(X, reachOptions); % Seems to be working

%% Example 7 --  mnist_fc
mnist_fc2_file = 'mnist_fc/onnx/mnist-net_256x2.onnx';
mnistOpts.InputDataFormat = "BTC'";
% mnistfc2 = onnx2nnv(vnnFolder+mnist_fc2_file, mnistOpts); % UPDATE matlab2nnv to remove useless layers in the beginning

%% Example 8 -- nn4sys
nn4sys_file = 'nn4sys/onnx/lindex.onnx';
nn4sys_lindex = onnx2nnv(vnnFolder+nn4sys_file); % good

nn4sys_deep_file = 'nn4sys/onnx/lindex_deep.onnx';
nn4sys_deep_lindex = onnx2nnv(vnnFolder+nn4sys_deep_file); % good

%% Example 9 -- oval21
oval_base_file = 'oval21/onnx/cifar_base_kw.onnx';
oval_base = onnx2nnv(vnnFolder+oval_base_file);

oval_deep_file = 'oval21/onnx/cifar_deep_kw.onnx';
oval_deep = onnx2nnv(vnnFolder+oval_deep_file);

%% Example 10 -- reach_prob_density
density_gcas_file = 'reach_prob_density/onnx/gcas.onnx';
% density_gcas = onnx2nnv(vnnFolder+density_gcas_file); % Connections error (check which layers are getting eliminated)

density_robot_file = 'reach_prob_density/onnx/robot.onnx';
% density_robot = onnx2nnv(vnnFolder+density_robot_file); % same error

%% Example 11 -- rl_benchmarks
rl_cartpole_file = 'rl_benchmarks/onnx/cartpole.onnx';
rl_cartpole_options.InputDataFormat = 'BC';
% rl_cartpole = onnx2nnv(vnnFolder+rl_cartpole_file, rl_cartpole_options); %flatten layer, can get rid off the first 3 layers and that would work

rl_dubinsrejoin_file = 'rl_benchmarks/onnx/dubinsrejoin.onnx';
rl_dubinsrejoin_options.InputDataFormat = 'BC';
rl_dubinsrejoin = onnx2nnv(vnnFolder+rl_dubinsrejoin_file, rl_dubinsrejoin_options);


%% Example 12 -- sri_resnet_a
sri_a_file = 'sri_resnet_a/onnx/resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx';
sri_a = onnx2nnv(vnnFolder + sri_a_file); % loads, but resnet, no support
% Just need to add support to addition layer

%% Example 13 -- sri_resnet_b
% This will be the same as previous one

%% Example 14 -- test



%% Example 15 -- tllverifybench
tll_1_file = 'tllverifybench/onnx/tllBench_n=2_N=M=16_m=1_instance_1_0.onnx';
tll_options.InputDataFormat = 'BC';
tll_options.OutputDataFormat = 'BC';
tll_options.FoldConstants = "deep";
tll_1 = onnx2nnv(vnnFolder + tll_1_file, tll_options);

%% Example 16 -- vggnet16_2022
vgg_file = 'vggnet16_2022/onnx/vgg16-7.onnx';
vgg_options.OutputDataFormat = 'BC';
vgg_options.FoldConstants = "deep";
vgg16 = onnx2nnv(vnnFolder+vgg_file, vgg_options); % two placeholder layers that we can easily eliminate
