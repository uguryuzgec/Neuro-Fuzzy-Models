% Load Data...

function data_output=LoadData()

	data_orj=xlsread('2014temmuz');
	data = data_orj;
	% Data Normalization
	data = (data_orj - min(data_orj))/(max(data_orj)- min(data_orj));
	data(isnan(data)) = 0;
	nSample=size(data,1);
		
	% Train Data
    pTrain=0.7;
    nTrain=round(pTrain*nSample);
	N = 3; M = 1;
	TrainInputs = lagmatrix(data(1:end),[N:-1:0]);
	TrainInputs(isnan(TrainInputs)) = 0; 
    TrainTargets = lagmatrix(data(1:end),-[1:M]);
	TrainTargets(isnan(TrainTargets)) = 0; 
	TrainInputs = TrainInputs(N+1:nTrain,:);
	TrainTargets = TrainTargets(N+1:nTrain,end);
    
    % Test Data
    TestInputs = lagmatrix(data(1:end),[N:-1:0]);
	TestInputs(isnan(TestInputs)) = 0; 
	TestTargets = lagmatrix(data(1:end),-[1:M]);
	TestTargets(isnan(TestTargets)) = 0; 
	TestInputs = TestInputs(nTrain+1:end-M,:);
	TestTargets = TestTargets(nTrain+1:end-M,end);
	     
    % Export Data
    data_output.TrainInputs=TrainInputs;
    data_output.TrainTargets=TrainTargets;
    data_output.TestInputs=TestInputs;
    data_output.TestTargets=TestTargets;

end