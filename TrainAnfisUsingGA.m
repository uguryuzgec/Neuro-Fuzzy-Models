function output=TrainAnfisUsingGA(fis,data)

    %% Problem Definition
    
    p0=GetFISParams(fis);
    Problem.CostFunction=@(x) TrainFISCost(x,fis,data);
    Problem.nVar=numel(p0);
    Problem.VarMin=-10*max(data.TrainInputs(:));
    Problem.VarMax=10*max(data.TrainInputs(:));

    %% GA Params
    Params.MaxIt = 1000;
    Params.nPop = round(10+2*sqrt(Problem.nVar));
	Params.refresh = 100;
	
    %% Run GA
    results=RunGA(Problem,Params);
    
    %% Get Results
    
    p=results.BestSol.Position.*p0;
    bestfis=SetFISParams(fis,p);
    output.bestfis=bestfis;
	output.bestcost=results.BestCost;
	output.bestsol=results.BestSol;    
end

function results=RunGA(Problem,Params)

    disp('Starting GA ...');
	refresh=Params.refresh;
	
    %% Problem Definition

    CostFunction=Problem.CostFunction;        % Cost Function

    nVar=Problem.nVar;          % Number of Decision Variables

    VarSize=[1 nVar];           % Size of Decision Variables Matrix

    VarMin=Problem.VarMin;      % Lower Bound of Variables
    VarMax=Problem.VarMax;      % Upper Bound of Variables

    %% GA Parameters

    MaxIt=Params.MaxIt;      % Maximum Number of Iterations

    nPop=Params.nPop;        % Population Size

    pc=0.4;                 % Crossover Percentage
    nc=2*round(pc*nPop/2);  % Number of Offsprings

    pm=0.7;                 % Mutation Percentage
    nm=round(pm*nPop);      % Number of Mutants

    gamma=0.7;

    mu=0.15;        % Mutation Rate

    beta=8;         % Selection Pressure

    %% Initialization

    empty_individual.Position=[];
    empty_individual.Cost=[];

    pop=repmat(empty_individual,nPop,1);

    for i=1:nPop

        % Initialize Position
        if i>1
            pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
        else
            pop(i).Position=ones(VarSize);
        end

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

    % Store Cost
    WorstCost=pop(end).Cost;


    %% Main Loop

    for it=1:MaxIt

        P=exp(-beta*Costs/WorstCost);
        P=P/sum(P);

        % Crossover
        popc=repmat(empty_individual,nc/2,2);
        for k=1:nc/2

            % Select Parents Indices
            i1=RouletteWheelSelection(P);
            i2=RouletteWheelSelection(P);

            % Select Parents
            p1=pop(i1);
            p2=pop(i2);

            % Apply Crossover
            [popc(k,1).Position, popc(k,2).Position]=...
                Crossover(p1.Position,p2.Position,gamma,VarMin,VarMax);

            % Evaluate Offsprings
            popc(k,1).Cost=CostFunction(popc(k,1).Position);
            popc(k,2).Cost=CostFunction(popc(k,2).Position);

        end
        popc=popc(:);


        % Mutation
        popm=repmat(empty_individual,nm,1);
        for k=1:nm

            % Select Parent
            i=randi([1 nPop]);
            p=pop(i);

            % Apply Mutation
            popm(k).Position=Mutate(p.Position,mu,VarMin,VarMax);

            % Evaluate Mutant
            popm(k).Cost=CostFunction(popm(k).Position);

        end

        % Create Merged Population
        pop=[pop 
             popc
             popm];  

        % Sort Population
        Costs=[pop.Cost];
        [Costs, SortOrder]=sort(Costs);
        pop=pop(SortOrder);

        % Update Worst Cost
        WorstCost=max(WorstCost,pop(end).Cost);

        % Truncation
        pop=pop(1:nPop);
        Costs=Costs(1:nPop);

        % Store Best Solution
        BestSol=pop(1);

        % Store Best Cost
        BestCost(it)=BestSol.Cost;

        % Show Iteration Information
		if (rem(it,refresh)==0)
			disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
		end
    end

    disp('End of GA.');
    disp(' ');
	
    %% Results

    results.BestSol=BestSol;
    results.BestCost=BestCost;
    
end

function [y1, y2]=Crossover(x1,x2,gamma,VarMin,VarMax)

    alpha=unifrnd(-gamma,1+gamma,size(x1));
    
    y1=alpha.*x1+(1-alpha).*x2;
    y2=alpha.*x2+(1-alpha).*x1;
    
    y1=max(y1,VarMin);
    y1=min(y1,VarMax);
    
    y2=max(y2,VarMin);
    y2=min(y2,VarMax);

end

function y=Mutate(x,mu,VarMin,VarMax)

    nVar=numel(x);
    
    nmu=ceil(mu*nVar);
    
    j=randsample(nVar,nmu)';
    
    sigma=0.1*(VarMax-VarMin);
    
    y=x;
    y(j)=x(j)+sigma*randn(size(j));
    
    y=max(y,VarMin);
    y=min(y,VarMax);

end