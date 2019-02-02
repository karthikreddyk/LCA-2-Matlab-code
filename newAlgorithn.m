clc;clear
%% system being considered
%A = [0.5, 0; 0, 0.9]; % originally stable system
%A = [1, 0; 0,1]; % Marginally stable system

A = [0.5, 0.1; 3,0.9]; % Unstable system
%A = A_small;
% 0.01*
Q = [2, 1; 1,4]; % considering an infinite horizon case
%Q = [2, 0; 0,4]; % +ve def matrix, no cross coupling
%Q = Q_small;
B1 = [1; 3]; % Agent 1 affine matrix
B2 = [2; 7]; % Agent 2 affine matrix
B = [B1, B2];
%B = B_small;
n = 2; N = 2; % number of states , number of agents
theta1 = 5; % for agent 1
theta2 = 15; % for agent 2
theta_max = 100;
theta_min = 0.1;


%% controllability, observability test for the chosen system
Co = ctrb(A,B);
if (rank(Co) == n)
    disp('the system is controllable');
elseif (rank(Co) ~= n)
    disp('the system is UNCONTROLLABLE');
end

% Assume that all states are accesible to the outside world, which ensures
%%

m = 100; % just a value within which R_est is assumed to converge to R
R = blkdiag(theta1, theta2); % true value of R
R_storage = zeros(m,2); % for storing values of thetas during evolution

Agent1_Rstorage = cell(m,1); % for Evolution of R at agent1 for agent1
Agent1_Rstorage{1} = blkdiag(1, 1); % Initial estimates of Agent 1 at Agent 1


Agent2_Rstorage = cell(m,1); % for Evolution of R at agent2 for agent2
Agent2_Rstorage{1} = blkdiag(1, 1); % Initial estimates of Agent 2 at Agent 2

Xint = [-100;-120];
X = Xint;
X_evolution = cell(m,1); % to capture state evolution
X_evolution{1} = Xint;
U_evolution = cell(m,1); % to capture control evolution
R1_control = blkdiag(theta1, 1); % for the control matrix at agent 1
R2_control = blkdiag(1, theta2); % for the control matrix at agent 2

%% looping starts here
i =  1; % introduce a loop here 
while i < (m+1)
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
    X_evolution{i} = (A * X_evolution{i-1}) + B1* u1_observed + B2*u2_observed;
%     if (i == 100)
%         X_evolution{i} = X_evolution{i} + [3000;5000];
%     end    
end

stateevol = zeros(m,2);

stepsForPlotting = 10;

 
for i = 1:m
     stateevol(i,:)= X_evolution{i};
 end 
 
U_norm = zeros(m,2);
for j = 1:m
    U_norm(j,:) = U_evolution{j};
end
%hold on;
%plot(U_norm);
%figure;
%plot(stateevol);
for (i = 1:m)
    R_storage(i,:) = [Agent1_Rstorage{i}(1,1),Agent1_Rstorage{i}(2,2)];
end


figure(2)
clf
subplot(311)
hold on
plot(R_storage(1:stepsForPlotting,1),'-.ob')
plot(R_storage(1:stepsForPlotting,2),'-.or')
title('Parameter Learning-LCA-2')
%xlabel('Iterations')
%ylabel('\textbf{Parameters $\theta_1$, $\theta_2$ evolution with iterations}','Interpreter','latex');
h=legend('$\theta_1$', '$\theta_2$')
set(h,'Interpreter','Latex');

%figure;
subplot(312)
hold on
plot(stateevol(1:stepsForPlotting,1),'-.ob')
plot(stateevol(1:stepsForPlotting,2),'-.or')
title('State evolution-LCA-2')
%xlabel('Iterations')
%ylabel('\textbf{States $x_{1}$, $x_{2}$ evolution}','Interpreter','latex');
h=legend('$x_1$', '$x_2$')
set(h,'Interpreter','Latex');
%hold off;

%figure;
subplot(313)
hold on
plot(U_norm(1:stepsForPlotting,1),'-.ob');
plot(U_norm(1:stepsForPlotting,2),'-.or');
title('Control inputs applied-LCA2')
%xlabel('Iterations')
%ylabel('Control Input $u_{1}$, $u_{2}$ evolution','Interpreter','latex');
l=legend('$u_1$', '$u_2$')
set(l,'Interpreter','Latex')
xlabel('Time step k');
