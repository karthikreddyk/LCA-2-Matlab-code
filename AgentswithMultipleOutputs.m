clc;clear
%% system being considered
%A = [0.5, 0; 0,0.9]; % originally stable system
A = [1, 3,2; 3,1,5;2,9,7]; % Unstable system
%A = [1, 0,0; 0,1,0;0,0,1]; % Unstable system
%A = [7, 1,-1; 2,3,0;7,3,3]; % Unstable system

%Q = [2, 0; 0,4]; % considering an infinite horizon case
%Q = 0.001*[20, 0, 0; 0, 30, 0; 0,0,50];
Q = [20, 0, 0; 0, 30, 0; 0,0,50]; % considering an infinite horizon case
B1 = [1,4; 3,5; 2,7]; % Agent 1 affine matrix
B2 = [2,5; 7,3; 1,2]; % Agent 2 affine matrix
B = [B1, B2];
n = 3; N = 2; % number of states , number of agents
theta1 = 50; % for agent 1
theta2 = 90; % for agent 2
R1 = theta1 * eye(2);
R2 = theta2 * eye(2);
%theta3 = 50; % for agent 3


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
% observability. Need to check with professor
%%
m = 500; % just a value within which R_est is assumed to converge to R
R = blkdiag(R1, R2); % true value of R

Agent1_Rstorage = cell(m,1); % for Evolution of R at agent1 for agent1
Agent1_Rstorage{1} = blkdiag(1*eye(2), 1 * eye(2)); % Initial estimates of Agent 1 at Agent 1
U_evolution = cell(m,1); % to capture control evolution


Agent2_Rstorage = cell(m,1); % for Evolution of R at agent2 for agent2
Agent2_Rstorage{1} = blkdiag(1*eye(2), 1 * eye(2)); % Initial estimates of Agent 2 at Agent 2

Xint = [-10;-12; -30];
X = Xint;
X_evolution = cell(m,1); % to capture state evolution
X_evolution{1} = Xint;
R1_control = blkdiag(theta1 * eye(2) , 1 * eye(2)); % for the control matrix at agent 1
R2_control = blkdiag(1 * eye(2), theta2 * eye(2)); % for the control matrix at agent 2

%% looping starts here
i =  1; % introduce a loop here 
while i < (m+1)
    [Sinf_ag1,L_ag1,G_ag1] = dare(A,B,Q,R1_control);
    [Sinf_ag2,L_ag2,G_ag2] = dare(A,B,Q,R2_control);
    U1_observed = -G_ag1 * X_evolution{i}; % 2 x 1 vector
    U2_observed = -G_ag2 * X_evolution{i}; 
    u1_observed = U1_observed(1:2);
    u2_observed = U2_observed(3:4);
    U_evolution{i} = [u1_observed; u2_observed];    
    
    %% Agent 1 learning theta1 
%    R is unused. So can be removed later.
    R1_theta1_update = agentlearningforMIMO(A,B,Q,R,Agent1_Rstorage{i},u1_observed,X_evolution{i},theta_max,theta_min, 1);

    %% Agent 1 learning theta2
    R1_theta2_update = agentlearningforMIMO(A,B,Q,R,Agent1_Rstorage{i},u2_observed,X_evolution{i},theta_max,theta_min, 2);

    %% Agent 2 learning theta1 
    R2_theta1_update = agentlearningforMIMO(A,B,Q,R,Agent2_Rstorage{i},u1_observed,X_evolution{i},theta_max,theta_min, 1);

    %% Agent 2 learning theta2
    R2_theta2_update = agentlearningforMIMO(A,B,Q,R,Agent2_Rstorage{i},u2_observed,X_evolution{i},theta_max,theta_min, 2);

    %% control R Updates and saving
    i = i+1;
    Agent1_Rstorage{i} = blkdiag(R1_theta1_update(1:2,1:2),R1_theta2_update(3:4,3:4));
    Agent2_Rstorage{i} = blkdiag(R2_theta1_update(1:2,1:2),R2_theta2_update(3:4,3:4));
    R1_control(3:4,3:4) = Agent1_Rstorage{i}(3:4,3:4);
    R2_control(1:2,1:2) = Agent2_Rstorage{i}(1:2,1:2);
    %% state update
    X_evolution{i} = (A * X_evolution{i-1}) + B * [u1_observed;u2_observed];
%     if (i == 100)
%         X_evolution{i} = X_evolution{i} + [300;500;200];
%     end        
end

%%

stateevol = zeros(m,3);

stepsForPlotting = 10;
 
for i = 1:m
     stateevol(i,:)= X_evolution{i};
 end 
 
U_norm = zeros(m,4);
for j = 1:(m -1)
    U_norm(j,:) = U_evolution{j};
end
%hold on;
%plot(U_norm);
%figure;
%plot(stateevol);
for (i = 1:(m-1))
    R_storage(i,:) = [Agent1_Rstorage{i}(1,1),Agent1_Rstorage{i}(4,4)];
end

figure(3)
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
plot(stateevol(1:stepsForPlotting,3),'-.og');
title('State evolution-LCA-2')
%xlabel('Iterations')
%ylabel('\textbf{States $x_{1}$, $x_{2}$ evolution}','Interpreter','latex');
h=legend('$x_1$','$x_2$', '$x_3$')
set(h,'Interpreter','Latex');
%hold off;

%figure;
subplot(313)
hold on
plot(U_norm(1:stepsForPlotting,1),'-.ob');
plot(U_norm(1:stepsForPlotting,2),'-.or');
plot(U_norm(1:stepsForPlotting,3),'-.og');
plot(U_norm(1:stepsForPlotting,4),'-.om');
title('Control inputs applied-LCA2')
%xlabel('Iterations')
%ylabel('Control Input $u_{1}$, $u_{2}$ evolution','Interpreter','latex');
l=legend('$u_{11}$', '$u_{12}$', '$u_{21}$', '$u_{22}$')
set(l,'Interpreter','Latex')
xlabel('Time step k');