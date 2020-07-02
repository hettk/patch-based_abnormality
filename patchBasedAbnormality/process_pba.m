% Main function to estimate patch-based abnormality map
%
% process_pba(input, templates, options):  Main function to estimate
% abnormality map given a control subject population (see Hett et al.
% MICCAI 2020)
% 
% Agurments: 
%   - input
%       input.mri : path of the mri understudy
%       input.mask: mask of the region of interest
%       input.age : age of the subject understudy at scan
%   - templates
%       templates.t1_path  : list of path of the template mri
%       templates.mask_path: list of path of the template mask
%       templates.ages     : Vector of ages of templates at scan
%   - options 
%       OPAL parameters (default)
%       options.rss      = 8;    Size of the search area
%       options.ni       = 5;    Number of iter max
%       options.np       = 20;   Number of patch extracted
%       options.pr       = 3;    Patch size
%       options.hasmutex = 1;    Enable mutex 
%
% Return:
%   - pbi: abnormality map (patch distance in terms of intensity
%   differences)
%   - pbs: mean spatial distance map of patches extracted using OPAL
%
%
% Author: Kilian Hett, kilianhett@vanderbilt.edu (Vanderbilt University) 


function [pbi, pbs] = process_pba(input, templates, options)
addpath('OPAL');

input_mri   = input.mri;
input_mask  = input.mask;
input_age   = input.age;

if nargin<3
    rss      = 8;       
    ni       = 5;       
    np       = 20;      
    pr       = 3;       
    hasmutex = 1;        
else
    rss      = options.rss;         
    ni       = options.ni;         
    np       = options.np;        
    pr       = options.pr;          
    hasmutex = options.hasmutex;   
end

% Data 
t1_path    = templates.t1_path;
mask_path  = templates.mask_path;
% Demographics
ages        = templates.ages;

N = 30; 


tic;
[~,sorted_idx] = sort(abs(input_age - ages));
for j=1:N
    I = niftiread(t1_path{sorted_idx(j)});
    M = niftiread(mask_path{sorted_idx(j)});

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
I = zeros(size(mask_u));
I(pr+1:end-pr-1,pr+1:end-pr-1,pr+1:end-pr-1) = niftiread(input_mri);

% Segmentation mask
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
    
end

