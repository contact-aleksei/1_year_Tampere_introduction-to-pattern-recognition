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

%TASK three TASK three TASK three TASK three TASK three TASK three
%normal distribution parameters
F=tr_data;
labels=tr_labels;
[mu,covar,p]=bayeslearn(F,labels);
%function that returns the Bayesian optimal class c for the sample f
%c = cifar_10_bayes_classify(f,mu,sigma,p);
cl=[];
for i=1:10000;
    x=te_data(i,1:3072);
    f=features(x);
    %function that returns the Bayesian optimal class c for the sample f
    c =cifar_10_multivariate_classify(f,mu,covar);
    cl=[cl c];
end;
comp = (cl'==te_labels);
s=sum(comp); %disp(s);
acc=s/numel(te_labels)*100;
fprintf('Accuracy is '); disp(acc); fprintf('percent ');   
 function f = features(x)
 R=x(1:1024); rm=mean(R);
 G=x(1025:2048); gm=mean(G);
 B=x(2049:3072); bm=mean(B);
 f=[rm,gm,bm];   
 end
function [mu,covar,p]=bayeslearn(F,labels)
F1 = [F labels];
BS = sortrows(F1,3073);
meanRed=[];meanGreen=[];meanBlue=[];
for i = 1:50000
      rowt=BS(i,1:3072);
      R=rowt(1:1024); meanR=mean(R);   meanRed=[meanRed meanR];
      G=rowt(1025:2048); meanG=mean(G);meanGreen=[meanGreen meanG];
      B=rowt(2049:3072);meanB=mean(B); meanBlue=[meanBlue meanB];
      %%50000 rows 3 columns
end
meanRed=meanRed'; meanGreen=meanGreen'; meanBlue=meanBlue';
muall=[meanRed,meanGreen,meanBlue];
meanclassesR=[]; meanclassesG=[];meanclassesB=[];meatr=[];meatg=[];meatb=[];
kk=[];co=[];covar=[];

for i=0:9
    meanclassesR=[]; meanclassesG=[];meanclassesB=[];
    meatr=muall((5000*i+1):(5000*(i+1)),1);
    meanclassesR=[meanclassesR meatr];
    meatg=muall((5000*i+1):(5000*(i+1)),2); 
    meanclassesG=[meanclassesG meatg];
    meatb=muall((5000*i+1):(5000*(i+1)),3);
    meanclassesB=[meanclassesB meatb];
    kk= [meanclassesR, meanclassesG, meanclassesB];
    
    %10 rows 3 columns
    co=cov(kk);
    covar(:,:,i+1)=co;
end
B = sortrows(F1,3073);
meanyR=[]; meanyG=[]; meanyB=[]; sigmaR=[];sigmaG=[];sigmaB=[];
BR0 = B(1:5000,1:1024);        
S0R=mean(BR0); m0r=mean(S0R);
std0r=std(S0R); meanyR=[meanyR m0r];
sigmaR=[sigmaR std0r];
BR1 = B(5001:10000,1:1024);    S1R=mean(BR1); m1r=mean(S1R);std1r=std(S1R);meanyR=[meanyR m1r];sigmaR=[sigmaR std1r];
BR2 = B(10001:15000,1:1024);   S2R=mean(BR2); m2r=mean(S2R);std2r=std(S2R);meanyR=[meanyR m2r];sigmaR=[sigmaR std2r];
BR3 = B(15001:20000,1:1024);    S3R=mean(BR3); m3r=mean(S3R);std3r=std(S3R);meanyR=[meanyR m3r];sigmaR=[sigmaR std3r];
BR4 = B(20001:25000,1:1024);   S4R=mean(BR4); m4r=mean(S4R);std4r=std(S4R);meanyR=[meanyR m4r];sigmaR=[sigmaR std4r];
BR5 = B(25001:30000,1:1024);   S5R=mean(BR5); m5r=mean(S5R);std5r=std(S5R);meanyR=[meanyR m5r];sigmaR=[sigmaR std5r];
BR6 = B(30001:35000,1:1024);   S6R=mean(BR6); m6r=mean(S6R);std6r=std(S6R);meanyR=[meanyR m6r];sigmaR=[sigmaR std6r];
BR7 = B(35001:40000,1:1024);   S7R=mean(BR7); m7r=mean(S7R);std7r=std(S7R);meanyR=[meanyR m7r];sigmaR=[sigmaR std7r];
BR8 = B(40001:45000,1:1024);   S8R=mean(BR8); m8r=mean(S8R);std8r=std(S8R);meanyR=[meanyR m8r];sigmaR=[sigmaR std8r];
BR9 = B(45001:50000,1:1024);  S9R=mean(BR9); m9r=mean(S9R); std9r=std(S9R);meanyR=[meanyR m9r];sigmaR=[sigmaR std9r];
meanyR=meanyR';sigmaR=sigmaR';
BG0 = B(1:5000,1025:2048);       S0G=mean(BG0); m0g=mean(S0G);std0g=std(S0G);meanyG=[meanyG m0g];sigmaG=[sigmaG std0g];
BG1 = B(5001:10000,1025:2048);   S1G=mean(BG1); m1g=mean(S1G);std1g=std(S1G);meanyG=[meanyG m1g];sigmaG=[sigmaG std1g];
BG2 = B(10001:15000,1025:2048);  S2G=mean(BG2); m2g=mean(S2G);std2g=std(S2G);meanyG=[meanyG m2g];sigmaG=[sigmaG std2g];
BG3 = B(15001:20000,1025:2048);  S3G=mean(BG3); m3g=mean(S3G);std3g=std(S3G);meanyG=[meanyG m3g];sigmaG=[sigmaG std3g];
BG4 = B(20001:25000,1025:2048);  S4G=mean(BG4); m4g=mean(S4G);std4g=std(S4G);meanyG=[meanyG m4g];sigmaG=[sigmaG std4g];
BG5 = B(25001:30000,1025:2048);  S5G=mean(BG5); m5g=mean(S5G);std5g=std(S5G);meanyG=[meanyG m5g];sigmaG=[sigmaG std5g];
BG6 = B(30001:35000,1025:2048);  S6G=mean(BG6); m6g=mean(S6G);std6g=std(S6G);meanyG=[meanyG m6g];sigmaG=[sigmaG std6g];
BG7 = B(35001:40000,1025:2048);  S7G=mean(BG7); m7g=mean(S7G);std7g=std(S7G);meanyG=[meanyG m7g];sigmaG=[sigmaG std7g];
BG8 = B(40001:45000,1025:2048);  S8G=mean(BG8); m8g=mean(S8G);std8g=std(S8G);meanyG=[meanyG m8g];sigmaG=[sigmaG std8g];
BG9 = B(45001:50000,1025:2048);  S9G=mean(BG9); m9g=mean(S9G);std9g=std(S9G);meanyG=[meanyG m9g];sigmaG=[sigmaG std9g];
meanyG=meanyG'; sigmaG=sigmaG';
BB0 = B(1:5000,2049:3072);       S0B=mean(BB0); m0b=mean(S0B);std0b=std(S0B);meanyB=[meanyB m0b];sigmaB=[sigmaB std0b];
BB1 = B(5001:10000,2049:3072);   S1B=mean(BB1); m1b=mean(S1B);std1b=std(S1B);meanyB=[meanyB m1b];sigmaB=[sigmaB std1b];
BB2 = B(10001:15000,2049:3072);  S2B=mean(BB2); m2b=mean(S2B);std2b=std(S2B);meanyB=[meanyB m2b];sigmaB=[sigmaB std2b];
BB3 = B(15001:20000,2049:3072);  S3B=mean(BB3); m3b=mean(S3B);std3b=std(S3B);meanyB=[meanyB m3b];sigmaB=[sigmaB std3b];
BB4 = B(20001:25000,2049:3072);  S4B=mean(BB4); m4b=mean(S4B);std4b=std(S4B);meanyB=[meanyB m4b];sigmaB=[sigmaB std4b];
BB5 = B(25001:30000,2049:3072);  S5B=mean(BB5); m5b=mean(S5B);std5b=std(S5B);meanyB=[meanyB m5b];sigmaB=[sigmaB std5b];
BB6 = B(30001:35000,2049:3072);  S6B=mean(BB6); m6b=mean(S6B);std6b=std(S6B);meanyB=[meanyB m6b];sigmaB=[sigmaB std6b];
BB7 = B(35001:40000,2049:3072);  S7B=mean(BB7); m7b=mean(S7B);std7b=std(S7B);meanyB=[meanyB m7b];sigmaB=[sigmaB std7b];
BB8 = B(40001:45000,2049:3072);  S8B=mean(BB8); m8b=mean(S8B);std8b=std(S8B);meanyB=[meanyB m8b];sigmaB=[sigmaB std8b];
BB9 = B(45001:50000,2049:3072);  S9B=mean(BB9); m9b=mean(S9B);std9b=std(S9B);meanyB=[meanyB m9b];sigmaB=[sigmaB std9b];
meanyB=meanyB';sigmaB=sigmaB';
mu=[meanyR,meanyG,meanyB];
%sigma=[sigmaR,sigmaG,sigmaB];
disp('MEAN VALUES');
disp('STANDARD DEVIATION');
p=[1/10;1/10;1/10;1/10;1/10;1/10;1/10;1/10;1/10;1/10];
end
function c =cifar_10_multivariate_classify(f,mu,covar)
c=mvnpdf(f,mu,covar);disp(c);[M,I]=max(c); c=I-1;
end