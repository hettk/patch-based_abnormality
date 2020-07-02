% pbd(mri, nn, nnd, mask, patch_radius):  Compute abnormality map using distance of most 
%       similar patches extracted using PatchMatch algorithm (see Hett et
%       al. MICCAI 2020)
% 
% Agurments: 
%   - mri : 3D image understudy 
%   - nn  : 6d array describing localization of most similar patches (x,y,z,t,n,ijk)
%   - nnf : 5d array describing intensity distances of most similar patches
%   (x,y,z,t,n)
%   - mask: Mask of the region of interest
%   - patch_radius: size of the patch used to detect similarity  
%
%
% Return:
%   - dImap: abnormality map (patch distance in terms of intensity
%   differences)
%   - dSmap: mean spatial distance map of patches extracted using OPAL
%
% Author: Kilian Hett, kilianhett@vanderbilt.edu (Vanderbilt University) 


function [dImap,dSmap] = pbd(mri, nn, nnd, mask, patch_radius)

dImap = zeros(size(mask));
dSmap = zeros(size(mask));
idx  = find(mask>0); 

for i=1:length(idx)
  [x,y,z] = ind2sub(size(mask), idx(i));
  
  % Estimation of total intensity variation over the patch
  Px = mri((x-patch_radius):(x+patch_radius),...
           (y-patch_radius):(y+patch_radius),...
           (z-patch_radius):(z+patch_radius));
  
  Idis = nnd(x,y,z,:) / std(Px(:)); 
  Xdis = double(x-reshape(nn(x,y,z,2,:), size(nn,5), 1)).^2;
  Ydis = double(y-reshape(nn(x,y,z,1,:), size(nn,5), 1)).^2;
  Zdis = double(z-reshape(nn(x,y,z,3,:), size(nn,5), 1)).^2;

  [valI,~] = sort(Idis(:));
  [valS,~] = sort(Xdis + Ydis + Zdis);

  dImap(x,y,z) = mean(sqrt(valI(1:end)));
  dSmap(x,y,z) = mean(sqrt(valS(1:end)));
end

end





