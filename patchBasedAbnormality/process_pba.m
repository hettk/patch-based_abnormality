%
%
% process_pba(input, templates, options):  Main function to estimate
% abnormality map given a control subject population (see Hett et al.
% MICCAI 2020)
%
% 
% Agurments: 
%   - input
%   - templates
%   - options
%
%
% Return:
%   - pbi: abnormality map (patch distance in terms of intensity
%   differences)
%   - pbs: mean spatial distance map of patches extracted using OPAL
%
% Author: Kilian Hett, kilianhett@vanderbilt.edu (Vanderbilt University) 



addpath('OPAL');

% % OPAL parameters
% rss      = 8;	    % Size of the search area
% ni       = 5;	    % Number of iter max
% np       = 20;	% Number of patch extracted
% pr       = 3;	    % Patch size
% hasmutex = 1;     % Enable mutex 



function [pbi, pbs] = process_pba(input, templates, options)

input_mri   = input.mri;
input_mask  = input.mask;
input_age   = input.age;

% OPAL parameters
rss      = options.rss;         % Size of the search area
ni       = options.ni;          % Number of iter max
np       = options.np;          % Number of patch extracted
pr       = options.pr;          % Patch size
hasmutex = options.hasmutex;    % Enable mutex 


% Input folder
% inpath = '/data/h_oguz_lab/hettk/data/source/MNI-PREDICT-HD';

% Data 
t1_path    = templates.t1_path;
mask_path  = templates.mask_path;

% Demographics
% id_subj     = templates.id_subj;
% id_visi     = templates.id_visi;
group       = templates.group;
ages        = templates.ages;

% cag         = templates.cag;

% fid     = fopen(sprintf('%s/demographics_3T.csv',inpath));
% data    = textscan(fid, '%s%s%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d', 'Delimiter', ',');
% id_subj = data{1};
% id_visi = data{2};
% group   = data{8};
% ages    = data{3};
% scan    = data{9};
% cag     = data{5};
% fclose(fid);

% Load pre-computed cv partition table
% load ('cn_partition');
cn_partition = options.cn_partition;

% Load training data
cn_idx  = find(group==0);
% hd_idx  = find(group~=0);

N = 30; % 1..length(cn_tr)

% cn_tr   = cn_idx(cn_partition==p);
cn_tr   = cn_idx(cn_partition);

% cn_te   = cn_idx(cn_partition~=p);

% -- For testing purpose -- random sampling of testing dataset
% [~,rand_hd_idx] = sort(randn(size(hd_idx)));
% [~,rand_cn_idx] = sort(randn(size(cn_te)));
% te_idx = [hd_idx(rand_hd_idx(1:10)); cn_te(rand_cn_idx(1:10))]; 
% te_idx = [hd_idx; cn_te]; 

% % Output folder
% primary_path = '/data/h_oguz_lab/hettk/data/results';
% root = 'PBD';
% n=p;
% path_out = sprintf('%s/%s_%d', primary_path, root, n);
% mkdir(path_out);

% for i=1:length(te_idx)
%     fprintf('%s\n', t1_path{te_idx(i)});

    tic;
    [~,sorted_idx] = sort(abs(input_age - ages(cn_tr)));
    for j=1:N
    	I = niftiread(t1_path{cn_tr(sorted_idx(j))});
    	M = niftiread(mask_path{cn_tr(sorted_idx(j))});
    
    	if j==1
            temp = zeros(size(I,1)+(2*pr)+1, size(I,2)+(2*pr)+1, size(I,3)+(2*pr)+1, N);	
            mask_u = zeros(size(M)+(2*pr)+1);	
    	end	
    	temp(pr+1:end-pr-1,pr+1:end-pr-1,pr+1:end-pr-1,j) = I;
        mask_u(pr+1:end-pr-1,pr+1:end-pr-1,pr+1:end-pr-1) = M + mask_u(pr+1:end-pr-1,pr+1:end-pr-1,pr+1:end-pr-1);
    end
    temp = reshape(temp, size(temp,1), size(temp,2), size(temp,3)*size(temp,4));    
    mask_u = mask_u>0;
    t = toc;
    fprintf('Training template : %f\n',t);

    % Load subject file
    % T1 MRI
%     filename = sprintf('%s/%s', inpath, t1_path{te_idx(i)});
    I = zeros(size(mask_u));
    I(pr+1:end-pr-1,pr+1:end-pr-1,pr+1:end-pr-1) = niftiread(input_mri);
    info = niftiinfo(filename);

    % Segmentation mask
%     filename = sprintf('%s/%s', inpath, mask_path{te_idx(i)});
    M = zeros(size(mask_u));
    M(pr+1:end-pr-1,pr+1:end-pr-1,pr+1:end-pr-1) = niftiread(input_mask); 
    [mask_pm, nb_vox] = get_path_from_mask(M>0, 3);
    mask_pm = uint8(mask_pm);

    % Patch extraction
    tic;
    [nnf,nnfd] = opal_list(single(I), single(temp), uint8(mask_u),...
		 	ni, np, pr, rss, mask_pm, nb_vox, hasmutex);
    t1 = toc;
    fprintf('PatchMatch : %.1f sec \n', t);

    % Compute dissimilarity index
    [pbi,pbs] = pbd(I, nnf, nnfd, M, pr);
    t2 = toc;
    fprintf('Dissimilarity index : %.1f sec \n', t2 - t1);
    
    % Crop results
    pbi = pbi(pr+1:end-pr-1,pr+1:end-pr-1,pr+1:end-pr-1);
    pbs = pbs(pr+1:end-pr-1,pr+1:end-pr-1,pr+1:end-pr-1);
    

%     % Store results
%     filename = sprintf('%s/pbi_%s-%s_%d_%d',path_out,id_subj{te_idx(i)},id_visi{te_idx(i)},...
%                     ages(te_idx(i)), cag(te_idx(i))); % -- For testing purpose --
%     niftiwrite(dimap(pr+1:end-pr-1,pr+1:end-pr-1,pr+1:end-pr-1),filename,info,'Compressed',true);
% 
%     filename = sprintf('%s/pbs_%s-%s_%d_%d',path_out,id_subj{te_idx(i)},id_visi{te_idx(i)},...
%                     ages(te_idx(i)), cag(te_idx(i))); % -- For testing purpose --
%     niftiwrite(dsmap(pr+1:end-pr-1,pr+1:end-pr-1,pr+1:end-pr-1),filename,info,'Compressed',true);
% end

end

