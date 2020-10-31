function output=TrainAnfisUsingABC(fis,data)

    %% Problem Definition
    
    p0=GetFISParams(fis);
    
    Problem.CostFunction=@(x) TrainFISCost(x,fis,data);
    
    Problem.nVar=numel(p0);
    
    Problem.VarMin=-max(data.TrainInputs(:));
    Problem.VarMax=max(data.TrainInputs(:));

    %% ABC Params
	Params.MaxIt = 1000;
    Params.nPop = round(10+2*sqrt(Problem.nVar));
	Params.refresh = 100;
	
    %% Run ABC
    results=RunABC(Problem,Params);
    
    %% Get Results
    
    p=results.BestSol.Position.*p0;
    bestfis=SetFISParams(fis,p);
    output.bestfis=bestfis;
	output.bestcost=results.BestCost;
	output.bestsol=results.BestSol;
end

function results=RunABC(Problem,Params)

    disp('Starting ABC ...');
	refresh=Params.refresh;
	
    %% Problem Definition

    CostFunction=Problem.CostFunction;        % Cost Function

    nVar=Problem.nVar;          % Number of Decision Variables

    VarSize=[1 nVar];           % Size of Decision Variables Matrix

    VarMin=Problem.VarMin;      % Lower Bound of Variables
    VarMax=Problem.VarMax;      % Upper Bound of Variables

    %% Parameters

    MaxIt=Params.MaxIt;      % Maximum Number of Iterations
    nPop=Params.nPop;        % Population Size
	nOnlooker=nPop;          % Number of Onlooker Bees
	L=round(0.6*nVar*nPop);  % Abandonment Limit Parameter (Trial Limit)
	a=1;                     % Acceleration Coefficient Upper Bound
	
%% Initialization

% Empty Bee Structure
empty_bee.Position=[];
empty_bee.Cost=[];

% Initialize Population Array
pop=repmat(empty_bee,nPop,1);

% Initialize Best Solution Ever Found
BestSol.Cost=inf;

% Create Initial Population
for i=1:nPop
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    pop(i).Cost=CostFunction(pop(i).Position);
    if pop(i).Cost<=BestSol.Cost
        BestSol=pop(i);
    end
end

% Abandonment Counter
C=zeros(nPop,1);

% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);
   
%% Main Loop
for it=1:MaxIt

	for i=1:nPop
        
        % Choose k randomly, not equal to i
        K=[1:i-1 i+1:nPop];
        k=K(randi([1 numel(K)]));
        
        % Define Acceleration Coeff.
        phi=a*unifrnd(-1,+1,VarSize);
        
        % New Bee Position
        newbee.Position=pop(i).Position+phi.*(pop(i).Position-pop(k).Position);
        newbee.Position = max(newbee.Position, VarMin);
		newbee.Position = min(newbee.Position, VarMax);
        % Evaluation
        newbee.Cost=CostFunction(newbee.Position);
        
        % Comparision
        if newbee.Cost<=pop(i).Cost
            pop(i)=newbee;
        else
            C(i)=C(i)+1;
        end
        
    end
    
    % Calculate Fitness Values and Selection Probabilities
    F=zeros(nPop,1);
    MeanCost = mean([pop.Cost]);
    for i=1:nPop
        F(i) = exp(-pop(i).Cost/MeanCost); % Convert Cost to Fitness
    end
    P=F/sum(F);
    
    % Onlooker Bees
    for m=1:nOnlooker
        
        % Select Source Site
        i=RouletteWheelSelection(P);
        
        % Choose k randomly, not equal to i
        K=[1:i-1 i+1:nPop];
        k=K(randi([1 numel(K)]));
        
        % Define Acceleration Coeff.
        phi=a*unifrnd(-1,+1,VarSize);
        
        % New Bee Position
        newbee.Position=pop(i).Position+phi.*(pop(i).Position-pop(k).Position);
        newbee.Position = max(newbee.Position, VarMin);
		newbee.Position = min(newbee.Position, VarMax);
        % Evaluation
        newbee.Cost=CostFunction(newbee.Position);
        
        % Comparision
        if newbee.Cost<=pop(i).Cost
            pop(i)=newbee;
        else
            C(i)=C(i)+1;
        end
        
    end
    
    % Scout Bees
    for i=1:nPop
        if C(i)>=L
            pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
            pop(i).Cost=CostFunction(pop(i).Position);
            C(i)=0;
        end
    end
    
    % Update Best Solution Ever Found
    for i=1:nPop
        if pop(i).Cost<=BestSol.Cost
            BestSol=pop(i);
        end
    end

	% Update Best Cost
    BestCost(it)=BestSol.Cost;

    % Show Iteration Information
	if (rem(it,refresh)==0)
		disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
	end
end

    disp('End of ABC.');
    disp(' ');
    
    %% Results

    results.BestSol=BestSol;
    results.BestCost=BestCost; 
end