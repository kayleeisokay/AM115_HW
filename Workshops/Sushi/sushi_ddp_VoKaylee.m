% SUSHI_DP Sushi Restaurant Model
% Discrete Dynamic Programming
fprintf('\nSUSHI_DP SUSHI RESTAURANT MODEL\n')
close all
rng(116);
%include the path to the dynamical programming library
cepath='/Users/kayleeyvo/Downloads/CompEcon-master4students/';
path([cepath 'CEtools;' cepath 'CEdemos'],path);

% ENTER MODEL PARAMETERS

% sushi price ($/lb) when sold
% price = 20; % default
price   = 60; % high profit margin                             
cost    = 10;                             % sushi cost ($/lb) when bought
holdcost= 2;                             % opportunity cost, cost of shelf space


% Big Chain PARAMETERS
% Lower each parameter because of economies of scale
% price = 15;  
% cost = 5;
% holdcost = 1;

%Note: if there is no holding cost, then as long as you can adjust the next
%day, there won't be waste. In that case, there are multiple optimal solutions.

T=20;                                     % how many days to run the simulation for
%Define the probability mass function of the demand (a discretiziation of
%the true probability distribution function).
demand=50+10*(-2:1:2);                    %The set of possible demand values

%big store demand
% demand=200+10*(-3:1:3);

prob=ones(size(demand));                  %Probability mass function, assumed uniform here

%A different PMF for big store demand
% prob=[1 2 3 10 3 2 1];             

prob=prob/sum(prob);                      %total probability is 1
% Construct the state space. Both states are needed to evolve the system
% and compute the profit
maxdemand=max(demand);
s1 = (0:10:maxdemand)';                   % The set of carryover from the day before
s2 = (0:10:maxdemand)';                   % The set of amounts sold in the day before
S  = gridmake(s1,s2);                     % combined state grid
n  = length(S);                           % total number of states

% Space of possible actions
purchase=(0:10:maxdemand);                % in 10-lb increments
m=numel(purchase);                        % number of actions

% Construct reward function, which is a nxm matrix in this case
f=price*S(:,2)*ones(1,m)-cost*ones(n,1)*purchase-holdcost*S(:,1)*ones(1,m);

%gamma < 0 - risk seeking
%gamma = 0 - risk neutral
%gamma > 0 - risk averse
gamma = 1; 
utility = @(profit) (profit.^(1 - gamma) - 1) / (1 - gamma)
f = utility(max(f, 1e-6));

% Construct state transition probability matrix
g = [];
for j=1:m %action
    g1=zeros(n,n); %for a particular action
    for i=1:n %state
        for k=1:numel(demand) %demand
            %demand less than carryover from the day before
            if(demand(k)<S(i,1)) 
                 %what you buy today will be carried over to tomorrow
                 %everything else spoils
                carryover=purchase(j);
                %amount sold equals demand
                sold=demand(k); 
            
            %demand greater than carryover 
            %but less than carryover+purchase
            elseif(demand(k)<S(i,1)+purchase(j))
                %leftover becomes carryover
                carryover=S(i,1)+purchase(j)-demand(k);
                sold=demand(k);
             
            %demand greater than inventory 
            else      
                %nothing to carry over
                carryover=0;   
                %sell all inventory
                sold=S(i,1)+purchase(j);
            end
             %find the index of the new state
            l=getindex([carryover sold],S);
            %add the probability of this outcome to the transition matrix
            g1(i,l)=g1(i,l)+prob(k); 
        end
    end
    %stack them up to build the full transition probability matrix
    g=[g;g1]; 
end
g=sparse(g); %the matrix is in general sparse. turn it into a sparse for more efficient computation
% Pack model structure
clear model
model.reward     = f;
model.transprob  = g;
model.discount   = 0.8; %infinite horizon problem requires a discount rate less than 1
model.horizon    = T;
model.vterm= price*S(:,2); %value of final state. fish tossed away so no holding cost 


% Solve the discrete dynamic programming problem
[J,u,pstar] = ddpsolve(model);
% J is the optimal reward of the different states at different times
% u is the optimal control (purchase policy)
% pstar is the optimal transition matrix
% reshape them to view them as a function of s1 and s2
% The following is an example for the first time.
itime=1; % time index
uu=reshape(u(:,itime),[numel(s1) numel(s2)]);
contourf(s1,s2,purchase(uu));colorbar
xlabel('Fish sold the day before (lb)');
ylabel('Fish carried over from the day before (lb)');
title('Optimal purchase amount (lb)');

% ---- Trajectories ----
% Define initial state: carryover = 0, sold = 0
s_init = [0, 0]; 

% Find the index of the initial state
[~, idx] = ismember(s_init, S, 'rows');

% Initialize state and action trajectories
% Each row is vector: [carryover, sold]
% include state at t=1 and final at t=T+1
state_traj = zeros(T+1, 2); 
% Actions taken at t=1 to t=T
action_traj = zeros(T, 1);  

state_traj(1,:) = s_init;   % Initial state at t=1

for t = 1:T
    % Get current state index
    [~, idx] = ismember(state_traj(t,:), S, 'rows');
    
    % Get optimal action given state and time
    action_traj(t) = purchase(u(idx,t));
    
    % Sample demand
    demand_sample = randsample(demand, 1, true, prob);
    
    % Update state: very similar to the code
    % for constructing transition matrix
    if demand_sample < state_traj(t,1)
        carryover = action_traj(t);
        sold = demand_sample;
    elseif demand_sample < state_traj(t,1) + action_traj(t)
        carryover = state_traj(t,1) + action_traj(t) - demand_sample;
        sold = demand_sample;
    else
        carryover = 0;
        sold = state_traj(t,1) + action_traj(t);
    end
    
    state_traj(t+1,:) = [carryover, sold];
end

% Trim the last state, only want T steps
state_traj = state_traj(1:T,:);


figure;
subplot(2, 1, 1);
hold on; 

% Plot the state trajectory
plot(1:T, state_traj(:, 1));
plot(1:T, state_traj(:, 2));
xlabel('Time (days)');
ylabel('Amount (lb)');
legend('Carryover', 'Sold');
title('State Trajectory');

% Plot the optimal purchase trajectory
subplot(2, 1, 2);
plot(1:T, action_traj);
xlabel('Time (days)');
ylabel('Purchase (lb)');
title('Optimal Purchase Trajectory');