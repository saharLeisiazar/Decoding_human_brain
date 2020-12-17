clc
clear

load('training_testing.mat');

Training_tot_x = cat(1,Training_x,Testing_x);
Training_tot_y = cat(1,Training_y,Testing_y);

SVMModel = fitcsvm(Training_tot_x,Training_tot_y,'Standardize',true,'KernelFunction','polynomial','KernelScale','auto');
CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel);
cross_val = (1-classLoss)*100

SVMModel_2 = fitcsvm(Training_x,Training_y,'Standardize',true,'KernelFunction','polynomial','KernelScale','auto');

label_train = predict(SVMModel_2,Training_x);
label_test = predict(SVMModel_2,Testing_x);

train_acc=0;
for k=1:length(label_train)
    if Training_y(k) == label_train(k)
        train_acc = train_acc+1;
    end
end
train_acc= (train_acc/length(label_train))*100;

test_acc=0;
for k=1:length(label_test)
    if Testing_y(k) == label_test(k)
        test_acc = test_acc+1;
    end
end
test_acc= (test_acc/length(label_test))*100

