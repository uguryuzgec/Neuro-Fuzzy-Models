function output=TrainAnfisUsingPSO(fis,data)

    %% Problem Definition
    
    p0=GetFISParams(fis);
    Problem.CostFunction=@(x) TrainFISCost(x,fis,data);
    Problem.nVar=numel(p0);
    Problem.VarMin=-10*max(data.TrainInputs(:));
    Problem.VarMax=10*max(data.TrainInputs(:));

    %% PSO Params
    Params.MaxIt = 1000;
    Params.nPop = round(10+2*sqrt(Problem.nVar));
	Params.refresh = 100;

    %% Run PSO
    results=RunPSO(Problem,Params);
    
    %% Get Results

    p=results.BestSol.Position.*p0;    
    bestfis=SetFISParams(fis,p);
    output.bestfis=bestfis;
	output.bestcost=results.BestCost;
	output.bestsol=results.BestSol;    
end

function results=RunPSO(Problem,Params)

    disp('Starting PSO ...');
	refresh=Params.refresh;

    %% Problem Definition

    CostFunction=Problem.CostFunction;        % Cost Function

    nVar=Problem.nVar;          % Number of Decision Variables

    VarSize=[1 nVar];           % Size of Decision Variables Matrix

    VarMin=Problem.VarMin;      % Lower Bound of Variables
    VarMax=Problem.VarMax;      % Upper Bound of Variables

    %% PSO Parameters

    MaxIt=Params.MaxIt;      % Maximum Number of Iterations

    nPop=Params.nPop;        % Population Size  

    w=1;            % Inertia Weight
    wdamp=0.99;     % Inertia Weight Damping Ratio
    c1=1;           % Personal Learning Coefficient
    c2=2;           % Global Learning Coefficient

    % Velocity Limits
    VelMax=0.1*(VarMax-VarMin);
    VelMin=-VelMax;

    %% Initialization

    empty_particle.Position=[];
    empty_particle.Cost=[];
    empty_particle.Velocity=[];
    empty_particle.Best.Position=[];
    empty_particle.Best.Cost=[];

    particle=repmat(empty_particle,nPop,1);

    BestSol.Cost=inf;

    for i=1:nPop

        % Initialize Position
       
        particle(i).Position=unifrnd(VarMin,VarMax,VarSize);        

        % Initialize Velocity
        particle(i).Velocity=zeros(VarSize);

        % Evaluation
        particle(i).Cost=CostFunction(particle(i).Position);

        % Update Personal Best
        particle(i).Best.Position=particle(i).Position;
        particle(i).Best.Cost=particle(i).Cost;

        % Update Global Best
        if particle(i).Best.Cost<BestSol.Cost
            BestSol=particle(i).Best;
        end

    end

    BestCost=zeros(MaxIt,1);

    %% Main Loop

    for it=1:MaxIt

        for i=1:nPop

            % Update Velocity
            particle(i).Velocity = w*particle(i).Velocity ...
                +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
                +c2*rand(VarSize).*(BestSol.Position-particle(i).Position);

            % Apply Velocity Limits
            particle(i).Velocity = max(particle(i).Velocity,VelMin);
            particle(i).Velocity = min(particle(i).Velocity,VelMax);

            % Update Position
            particle(i).Position = particle(i).Position + particle(i).Velocity;

            % Velocity Mirror Effect
            IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
            particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);

            % Apply Position Limits
            particle(i).Position = max(particle(i).Position,VarMin);
            particle(i).Position = min(particle(i).Position,VarMax);

            % Evaluation
            particle(i).Cost = CostFunction(particle(i).Position);

            % Update Personal Best
            if particle(i).Cost<particle(i).Best.Cost

                particle(i).Best.Position=particle(i).Position;
                particle(i).Best.Cost=particle(i).Cost;

                % Update Global Best
                if particle(i).Best.Cost<BestSol.Cost
                    BestSol=particle(i).Best;
                end

            end

        end

        BestCost(it)=BestSol.Cost;

        % Show Iteration Information
		if (rem(it,refresh)==0)
			disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
		end

        w=w*wdamp;

    end

    disp('End of PSO.');
    disp(' ');

    %% Results

    results.BestSol=BestSol;
    results.BestCost=BestCost;
    
end
