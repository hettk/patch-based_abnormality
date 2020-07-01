%get_path_from_mask.m
% 
% Generates for 2D dimensions, the snail path coordinates. For 3D processing, the snail path
% is the same for each slice (z dim).
% The (x,y,z) coordinates to address are successively stored in 'snail_path' and
% snail_prop_offset registers the offsets to apply during the propagation
% step (on x and y, z offset being always -1 or +1 according to the PM iteration)


function [mask_path, nb_vox_mask] = get_path_from_mask(mask, pr)

[h,w,d] = size(mask);

nb_vox_mask = length(find(mask(1+pr:h-pr,1+pr:w-pr,1+pr:d-pr)>0));

count = 1;
mask_path = zeros(nb_vox_mask,3);
for z=1+pr:d-pr
   for y=1+pr:h-pr
       for x=1+pr:w-pr
           if (mask(y,x,z) > 0)
               mask_path(count, :) = [x y z];
               count = count + 1;
           end
            
        end
    end 
end

mask_path = mask_path-1;   %C-MEX
nb_vox_mask = size(mask_path,1);

end