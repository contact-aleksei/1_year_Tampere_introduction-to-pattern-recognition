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

%pred = [0:9];
%gt = [0:8 11];
%[acc]=cifar_ev(pred,gt);


x_data=te_data;
x_labels=te_labels;
[x]=cifar_randclas(x_labels);

pred = x;
gt = te_labels;
[acc]=cifar_ev(pred,gt);

function[acc]=cifar_ev(pred,gt)
comp = (pred==gt);
%disp(comp);
s=sum(comp);
acc=s/numel(gt)*100;
fprintf('sum of elements');
disp(s);
fprintf('Accuracy is '); disp(acc); fprintf('percent ');
end
function[x]=cifar_randclas(x_labels)
n=numel(x_labels);
x = randi(10,n,1)-1;
disp(x);
end

