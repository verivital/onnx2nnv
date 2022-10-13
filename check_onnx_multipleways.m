function check_onnx_multipleways()
    
    % Access the folder where vnncomp benchmarks are
    vnn_folder = strcat('../vnncomp2022_benchmarks/benchmarks');
    listing = dir(vnn_folder);
    folders = {listing.name};

    % Skip first 2 folders and iterate trhough the rest of them looking for onnx networks
    warning('off','all');
    diary vnncomp_trials.txt
    t_total = tic;
    for onnx_folder = folders(3:end)
        onnx_nets = dir([vnn_folder filesep onnx_folder{1} filesep 'onnx']);
        disp(onnx_folder{1});
        % for each file path attempt both importONNXLayer and importONNXNetwork
        t = tic;
        for net = 1:length(onnx_nets)
            if contains(onnx_nets(net).name,'.onnx')
                disp(onnx_nets(net).name);
                results = load_onnx([onnx_nets(net).folder filesep onnx_nets(net).name]);
                disp(results);
            end
        end
        toc(t);
        disp('------------------------------------------------------------')
    end
    toc(t_total);
    diary off;
    warning('on','all');
end

