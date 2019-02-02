clc;clear
%% system being considered
%A = [0.5, 0; 0,0.9]; % originally stable system
%A = [0.5, 0.1; 3,0.9]; % Unstable system
% A = Big_sys_Discrete.A;
% B = Big_sys_Discrete.B;
% C = Big_sys_Discrete.C;
% D = Big_sys_Discrete.D;
% Q = (Big_sys_Discrete.c)'*(Big_sys_Discrete.c);
% load('Big_sys_Discrete');
%load('Discrete_small_sys');

A = [[0.8804,0.0];[0.0,0.8804]];
B = [[0.009884, 0.0];[0.0, 0.009884]];
C = [[0.1001,  0.1001];[ -0.2003, 0.2003]];
D = [[0.0,0.0];[0.0,0.0]];

% A = Discrete_small_sys.A;
% B = Discrete_small_sys.B;
% C = Discrete_small_sys.C;
% D = Discrete_small_sys.D;
%Q = 100*(Discrete_small_sys.c)'*(Discrete_small_sys.c);
Q = 100*(C)'*(C);



%A = A_small;
%Q = [2, 0; 0,4]; % considering an infinite horizon case
%Q = [2, 1; 1,4]; % considering an infinite horizon case

%B1 = [1; 3]; % Agent 1 affine matrix
%B2 = [2; 7]; % Agent 2 affine matrix
%B = [B1, B2];
%B = B_small;
%C = C_small;
%Q = 10* (C_small' * C_small);
n = 2; N = 2; % number of states , number of agents

%% controllability, observability test for the chosen system
Co = ctrb(A,B);
if (rank(Co) == n)
    display('the system is controllable');
elseif (rank(Co) ~= n)
    display('the system is UNCONTROLLABLE');
end

Ob = obsv(A,C);
if (rank(Ob) == n)
    display('the system is Observable');
elseif (rank(Co) ~= n)
    display('the system is UNOBSERVABLE');
end

theta1 = 0.6; % for agent 1
theta2 = 0.3; % for agent 2
theta_max = 100;
theta_min = 0.001;


%%
m = 1500; % just a value within which R_est is assumed to converge to R
R = blkdiag(theta1, theta2); % true value of R

Agent1_Rstorage = cell(m,1); % for Evolution of R at agent1 for agent1
Agent1_Rstorage{1} = blkdiag(1, 1); % Initial estimates of Agent 1 at Agent 1


Agent2_Rstorage = cell(m,1); % for Evolution of R at agent2 for agent2
Agent2_Rstorage{1} = blkdiag(1, 1); % Initial estimates of Agent 2 at Agent 2

%Xint = [-10;-12];
%Xint = [-10;-12;-20;-19;-13;-7];
Yint = [-60;0];
%Xint = [-100;-100];
Cinv = inv(C);
Xint = C\Yint;
X = Xint;
X_evolution = cell(m,1); % to capture state evolution
X_evolution{1} = Xint;
U_evolution = cell(m,1); % to capture control evolution
R1_control = blkdiag(theta1, 1); % for the control matrix at agent 1
R2_control = blkdiag(1, theta2); % for the control matrix at agent 2

%% looping starts here
i =  1; % introduce a loop here 
while i < m
    [Sinf_ag1,L_ag1,G_ag1] = dare(A,B,Q,R1_control);
    [Sinf_ag2,L_ag2,G_ag2] = dare(A,B,Q,R2_control);
    U1_observed = -G_ag1 * X_evolution{i}; % 2 x 1 vector
    U2_observed = -G_ag2 * X_evolution{i}; 
    u1_observed = U1_observed(1);
    u2_observed = U2_observed(2);
    U_evolution{i} = [u1_observed; u2_observed];        
        
    %% Agent 1 learning theta1 
    % R is unused. So can be removed later.
    R1_theta1_update = agentlearning(A,B,Q,R,Agent1_Rstorage{i},u1_observed,X_evolution{i},theta_max,theta_min, 1);

    %% Agent 1 learning theta2
    R1_theta2_update = agentlearning(A,B,Q,R,Agent1_Rstorage{i},u2_observed,X_evolution{i},theta_max,theta_min, 2);

    %% Agent 2 learning theta1 
    R2_theta1_update = agentlearning(A,B,Q,R,Agent2_Rstorage{i},u1_observed,X_evolution{i},theta_max,theta_min, 1);

    %% Agent 2 learning theta2
    R2_theta2_update = agentlearning(A,B,Q,R,Agent2_Rstorage{i},u2_observed,X_evolution{i},theta_max,theta_min, 2);

    %% control R Updates and saving
    i = i+1;
    Agent1_Rstorage{i} = blkdiag(R1_theta1_update(1,1),R1_theta2_update(2,2));
    Agent2_Rstorage{i} = blkdiag(R2_theta1_update(1,1),R2_theta2_update(2,2));
    R1_control(2,2) = Agent1_Rstorage{i}(2,2);
    R2_control(1,1) = Agent2_Rstorage{i}(1,1);
    %% state update
    %X_evolution{i} = (A * X_evolution{i-1}) + B1* u1_observed + B2*u2_observed;
    if (i >= 100) && (i <= 250)            
        Disturbance = [-50;0];
%        disp(Disturbance);
    else
        Disturbance = [0;0];
    end
    X_evolution{i} = (A * X_evolution{i-1}) + B * [u1_observed ; u2_observed] + B * Disturbance;

end

%% Plotting

stateevol = zeros(m,2);
outputevol = zeros(m,2);

stepsForPlotting = 500;

 
for i = 1:m
     stateevol(i,:)= X_evolution{i};
     outputevol(i,:)= C*X_evolution{i};
end 
 



U_norm = zeros(m,2);
for j = 1:(m-1)
    U_norm(j,:) = U_evolution{j};
end

%%  for data from experiment plotting: Uncomment when plotting experiment data

state1 = states(:,1,1);
state2 = states(:,1,2);
expStates = [state1,state2];
stateevol = expStates;
%expOutputs = C * (expStates');
outputevol = [Outputs(:,1,1), Outputs(:,1,2)];



stepsForPlotting = 1500;
hold on;
%plot(U_norm);
%figure;
%plot(stateevol);
for (i = 1:(m-1))
    R_storage(i,:) = [Agent1_Rstorage{i}(1,1),Agent1_Rstorage{i}(2,2)];
end
figure;
plot(R_storage(1:stepsForPlotting,1),'-.ob');
hold on;
plot(R_storage(1:stepsForPlotting,2),'-.or');
title('Parameter Learning ')
xlabel('Iterations')
ylabel('Parameters $\theta_1$, $\theta_2$ evolution with iterations','Interpreter','latex');
legend({'$\theta_1$', '$\theta_2$'},'Interpreter','latex')
hold off;


figure;
plot(stateevol(1:stepsForPlotting,1),'-.ob');
%plot(state1(1:stepsForPlotting,1),'-.ob');
hold on;
plot(stateevol(1:stepsForPlotting,2),'-.or');
title('State evolution - Application of disturbance and recovery - LCA-2 Simulation')
xlabel('Iterations')
ylabel('States $x_{1}$, $x_{2}$ evolution','Interpreter','latex');
legend({'$x_1$', '$x_2$'},'Interpreter','latex')
hold off;

figure;
plot(outputevol(1:stepsForPlotting,1),'-.ob');
hold on;
plot(outputevol(1:stepsForPlotting,2),'-.or');
title('Output evolution - Application of disturbance and recovery - LCA-2')
xlabel('Iterations')
ylabel('Outputs $y_{1}$, $y_{2}$ evolution','Interpreter','latex');
legend({'$y_1$', '$y_2$'},'Interpreter','latex')
hold off;




figure;
plot(U_norm(1:stepsForPlotting,1),'-.ob');
hold on;
plot(U_norm(1:stepsForPlotting,2),'-.or');
title('Control inputs applied')
xlabel('Iterations')
ylabel('Control Input $u_{1}$, $u_{2}$ evolution','Interpreter','latex');
legend({'$u_1$', '$u_2$'},'Interpreter','latex')


