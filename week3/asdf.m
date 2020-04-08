clear; clc;
if exist('tr_data','var') && size(tr_data,1) == 50000
  disp('Seems that data exists, clean tr_data to re-read!');
  return;
end

conf.cifar10_dir = 'worktemp\cifar-10-batches-mat';
conf.train_files = {'data_batch_1.mat',...
                    'data_batch_2.mat',...
                    'data_batch_3.mat',...
                    'data_batch_4.mat',...
                    'data_batch_5.mat'};
conf.test_file = 'test_batch.mat';
conf.meta_file = 'batches.meta.mat';

load(fullfile(conf.cifar10_dir,conf.meta_file));

% Read TRAINING DATA and form the feature matrix and target output
tr_data = [];
tr_labels = [];
fprintf('Reading training data...\n');
for train_file_ind = 1:length(conf.train_files)
  fprintf('\r  Reading %s', conf.train_files{train_file_ind});
  load(fullfile(conf.cifar10_dir,conf.train_files{train_file_ind}));
  tr_data = [tr_data; data];
  tr_labels = [tr_labels; labels];
  
end;


fprintf('Done!\n');

% Read TEST DATA and form the feature matrix and target output
fprintf('Reading and showing test data...\n');
load(fullfile(conf.cifar10_dir,conf.test_file));
te_data = data;
te_labels = labels;

%TASK THREE  TASK THREE  TASK THREE  TASK THREE  TASK THREE  TASK THREE  
%normal distribution parameters
F=tr_data;
labels=tr_labels;


F1 = [F labels];
B = sortrows(F1,3073);
means12=[];mu=[];
for i = 1:50000
    rowt=F1(i,1:3072);
    meansrowt12=[];
    for k=0:11
        m=mean(rowt(256*k+1:256*(k+1)));
        meansrowt12=[meansrowt12 m];
        
        co=cov(meansrowt12);
        covar(:,:,k+1)=co;
    end
    means12=[means12; meansrowt12];
end