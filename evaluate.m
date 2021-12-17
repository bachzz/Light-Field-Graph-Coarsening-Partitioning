%LF1 = uint8(255*RGB); % Friends RGB - float
LF1 = uint8(RGB); % Vincent RGB - integer
LF2 = permute(Rec_reduced_LFlist,[2 3 4 5 1]);

s = size(LF1);
max_qual = 0;
max_i_j = [0,0];

for i=1:s(1),
    for j=1:s(1),
        qual = psnr(LF1(i,j,:,:,1),LF2(i,j,:,:));
        if qual >= max_qual,
            max_qual = qual;
            max_i_j(1) = i;
            max_i_j(2) = j;
        end
    end
end

disp(max_qual)
disp(max_i_j)