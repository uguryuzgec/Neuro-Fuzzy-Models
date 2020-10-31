function output=TrainAnfisUsingDE(fis,data)

    %% Problem Definition
    
    p0=GetFISParams(fis);
    Problem.CostFunction=@(x) TrainFISCost(x,fis,data);
    Problem.nVar=numel(p0);
    Problem.VarMin=-10*max(data.TrainInputs(:));
    Problem.VarMax=10*max(data.TrainInputs(:));

    %% DE Params
	Params.MaxIt = 1000;
    Params.nPop = round(10+2*sqrt(Problem.nVar));
	Params.refresh = 100;
	
    %% Run DE
    results=RunDE(Problem,Params);
    
    %% Get Results
    p=results.BestSol.Position.*p0;
    bestfis=SetFISParams(fis,p);
    output.bestfis=bestfis;
	output.bestcost=results.BestCost;
	output.bestsol=results.BestSol;
end

function results=RunDE(Problem,Params)

    disp('Starting DE ...');
	refresh=Params.refresh;
	
    %% Problem Definition

    CostFunction=Problem.CostFunction;        % Cost Function

    nVar=Problem.nVar;          % Number of Decision Variables

    VarSize=[1 nVar];           % Size of Decision Variables Matrix

    VarMin=Problem.VarMin;      % Lower Bound of Variables
    VarMax=Problem.VarMax;      % Upper Bound of Variables

    %% DE Parameters

    MaxIt=Params.MaxIt;      % Maximum Number of Iterations

    nPop=Params.nPop;        % Population Size

    F=0.8;
    CR=0.5; 
	beta_min=0.2;   % Lower Bound of Scaling Factor
	beta_max=0.8;   % Upper Bound of Scaling Factor

    %% Initialization

    empty_individual.Position=[];
    empty_individual.Cost=[];

    pop=repmat(empty_individual,nPop,1);

    for i=1:nPop

        % Initialize Position
        pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
        
        % Evaluation
        pop(i).Cost=CostFunction(pop(i).Position);

    end

    % Sort Population
    Costs=[pop.Cost];
    [Costs, SortOrder]=sort(Costs);
    pop=pop(SortOrder);

    % Store Best Solution
    BestSol=pop(1);

    % Array to Hold Best Cost Values
    BestCost=zeros(MaxIt,1);
  
%% Main Loop
for it=1:MaxIt

	for i=1:nPop
        
        x=pop(i).Position;
        
		A=randperm(nPop);
        
        A(A==i)=[];
        
        a1=A(1);
        a2=A(2);
        a3=A(3);       
          
        % Mutation
        % beta=unifrnd(beta_min,beta_max);
        beta=unifrnd(beta_min,beta_max,VarSize);

        y=pop(a1).Position+beta.*(pop(a2).Position-pop(a3).Position); % DE/rand/1/bin
        y = max(y, VarMin);
		y = min(y, VarMax);
		
        % Crossover
        z=zeros(size(x));
        j0=randi([1 numel(x)]);
        for j=1:numel(x)
            if j==j0 || rand<=CR
                z(j)=y(j);
            else
                z(j)=x(j);
            end
        end
        
        NewSol.Position=z;
        NewSol.Cost=CostFunction(NewSol.Position);
        
        if NewSol.Cost<pop(i).Cost
            pop(i)=NewSol;
            
            if pop(i).Cost<BestSol.Cost
               BestSol=pop(i);
            end
        end
        
    end

	% Update Best Cost
    BestCost(it)=BestSol.Cost;

    % Show Iteration Information
	if (rem(it,refresh)==0)
		disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
	end
end

    disp('End of DE.');
    disp(' ');
	
    %% Results

    results.BestSol=BestSol;
    results.BestCost=BestCost; 
end