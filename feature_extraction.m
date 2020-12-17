
clear
clc

m = 20;     %number of features to select
ratio = 0.8;   %proportion of training to all data
 
input = [];
output = [];

for sub = 1:16   %16 subject
    path = 'data/';  
    filename = sprintf(strcat(path,'train_subject%02d.mat'),sub);
    disp(strcat('Loading ',filename));
    data = load(filename);
    X= data.X;
    y= data.y;

    fp1 = (2*5)/250;
    fp2 = (2*6)/250;
    b = fir1(1,[fp1 fp2]);
    X = filter(b,1,X,[],3);
    
    covFaceAve = zeros(size(squeeze(X(1,:,126:end)) * squeeze(X(1,:,126:end))'));
    covScrAve = zeros(size(squeeze(X(1,:,126:end)) * squeeze(X(1,:,126:end))'));
    nTrials = size(X,1);
    nFace = 0;
    nScr = 0;

    for i=1:nTrials
        E = squeeze(X(i,:,126:end));
        tr = trace(E * E');
        if (y(i)==1)
            nFace = nFace +1;
            covFaceAve = covFaceAve + (E * E' / tr);
        else
           nScr = nScr +1;
           covScrAve = covScrAve + (E * E' / tr);
        end
    end

    covFaceAve = covFaceAve / nFace;
    covScrAve = covScrAve / nScr;

    covComp = covFaceAve + covScrAve;
    [uC,lambdaC] = eig(covComp);
    P = sqrt(pinv(lambdaC)) * uC';
    wightenedcovFaceAve = P * covFaceAve * P';
    [B,lambdaFaceAve] = eig(wightenedcovFaceAve);

    Z = zeros(size(X(:,:,126:end)));
    for i=1:nTrials
       Z(i,:,:) = (P'*B)' * squeeze(X(i,:,126:end));
    end

    %Selecting the first and last m rows as the features
    f = zeros(size(Z,1),2*m);
    for i=1:nTrials
        ZnZ = squeeze(Z(i,:,:));
        %Only keeping the non-zero columns and rows of Z
        ZnZ( ~any(ZnZ,2), : ) = [];  %rows
        ZnZ( :, ~any(ZnZ,1) ) = [];  %columns
        for j=1:2*m
            if (j<=m)
                f(i,j) = var(ZnZ(j,:));
            else
                f(i,j) = var(ZnZ(end-2*m+j,:));
            end
        end
        f(i,:) = log2 (f (i,:) / sum (f (i,:)));
    end
    
    input = [input ; f];
    output = [output ; y];
end

s = size(output,1);
idx = randperm(s)  ;
Training_x = input(idx(1:round(ratio*s)),:); 
Testing_x = input(idx(round(ratio*s)+1:end),:) ;

Training_y = output(idx(1:round(ratio*s)),1); 
Testing_y = output(idx(round(ratio*s)+1:end),:,:) ;
