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
[mu,covar,p]=bayeslearn(F,labels);

cl=[];
for i=1:10000;
    x=te_data(i,1:3072);
    f=features(x);
    %function that returns the Bayesian optimal class c for the sample f
    c =cifar_10_multivariate_classify(f,mu,covar);
    cl=[cl c];
end;
comp = (cl==te_labels');
s=sum(comp);
acc=s/numel(te_labels)*100;
fprintf('Accuracy is '); disp(acc); fprintf('percent ');
 function f = features(x)
 f=[];
 for k=0:11
     m=mean(x(256*k+1:256*(k+1)));
     f=[f m];
 end
 end           
function [mu,covar,p]=bayeslearn(F,labels)
F1 = [F labels];
B = sortrows(F1,3073);
means12=[];mu=[];
for i = 1:50000
    rowt=B(i,1:3072);
    meansrowt12=[];
    for k=0:11
        m=mean(rowt(256*k+1:256*(k+1)));
        meansrowt12=[meansrowt12 m];
    end
    means12=[means12; meansrowt12];
end
for i=1:12
    mu=[mu mean(means12(1:50000,i))];
end
 for i=0:9
 
 co=cov(means12((5000*i+1):(5000*(i+1)),:));
 covar(:,:,i+1)=co;
 end

p=zeros(1,10);p=p+0.1;
end


function c =cifar_10_multivariate_classify(f,mu,covar)
c=mvnpdf(f,mu,covar); [M,I]=max(c); c=I-1;
end