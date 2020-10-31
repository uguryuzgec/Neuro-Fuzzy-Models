%																		  %
%       Hybrid Neuro-Fuzzy Models based Meta-Heuristic Algorithms         %
%                    Source code demo version 1.0                         %
%                                                                         %
%                                                                         %
%       Author and programmer: Emrah DOKUR, Ugur YUZGEC, Mehmet KURBAN    %
%                                                                         %
%         e-Mail: uyuzgec@gmail.com                       		          %
%                 ugur.yuzgec@bilecik.edu.tr			                  %
%                                                                         %
% Paper: Emrah DOKUR, Ugur YUZGEC, Mehmet KURBAN        				  %
% Performance Comparison of Hybrid Neuro-Fuzzy Models using 			  %
% Meta-Heuristic Algorithms for Short-Term Wind Speed Forecasting		  %
%																		  %

clc;
clear;
close all;

%% Load Data
data=LoadData();

%% Generate Basic FIS
% ncluster = 2*size(data.TrainInputs,2);
  ncluster = 2;
  
%% Training Using PSO, GA, DE, ABC ...

Options = {'Genetic Algorithm', 'Particle Swarm Optimization','Differential Evolution Algorithm', 'Artificial Bee Colony Algorithm'};

[Selection, Ok] = listdlg('PromptString', 'Select training method for ANFIS:', ...
                          'SelectionMode', 'single', ...
                          'ListString', Options);

pause(0.01);
          
if Ok==0
    return;
end

switch Selection
    case 1, fis=CreateInitialFIS(data,ncluster);
            output=TrainAnfisUsingGA(fis,data);  str=string('GA'); 
            bestcost=output.bestcost;
			fis=output.bestfis;      
    case 2, fis=CreateInitialFIS(data,ncluster);
            output=TrainAnfisUsingPSO(fis,data); str=string('PSO');
            bestcost=output.bestcost;
			fis=output.bestfis;     
    case 3, fis=CreateInitialFIS(data,ncluster);
            output=TrainAnfisUsingDE(fis,data);  str=string('DE');
            bestcost=output.bestcost;
			fis=output.bestfis;     
    case 4, fis=CreateInitialFIS(data,ncluster);
            output=TrainAnfisUsingABC(fis,data); str=string('ABC');
            bestcost=output.bestcost;
			fis=output.bestfis;    
end

    figure
	[x,mf] = plotmf(fis,'input',1);
	subplot(2,1,1), plot(x,mf)
	title(string('training using ')+str)
	xlabel('Membership Functions for Input 1')
	[x,mf] = plotmf(fis,'input',2);
	subplot(2,1,2), plot(x,mf)
	xlabel('Membership Functions for Input 2')

	figure
    plot(bestcost);
	xlabel('iteration')
	ylabel('RMSE')
	title(string('training using ')+str)

%% Results

% Train Data
TrainOutputs=evalfis(data.TrainInputs,fis);
PlotResults(data.TrainTargets,TrainOutputs,string('Training Result by ')+str);

% Test Data
TestOutputs=evalfis(data.TestInputs,fis);
PlotResults(data.TestTargets,TestOutputs,string('Test Result by ')+str);