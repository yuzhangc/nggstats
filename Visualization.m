% Data Visualization
% Yuzhang Chen
% 14 March 2020

%% Load And Extract Values

load('ExampleSims4YZ.mat')
choice_data = SimulationExamples.choice_data;
location_Rh_data = SimulationExamples.location_Rh_data;
location_Rl_data = SimulationExamples.location_Rl_data;
Hused = SimulationExamples.Hused;
change_data = SimulationExamples.change_data;
reward = SimulationExamples.reward;
Rh_data = SimulationExamples.Rh_data;
Rl_data = SimulationExamples.Rl_data;
RH_MU = 3;
RL_MU = -3;
SIGMA_H = 2;
SIGMA_L = 2;

% Determine if Changes Occurred

switch_choice = choice_data(2:1000,:) ~= choice_data(1:999,:);

% Segregate Data According to Hazard Rate

for i = 1:size(change_data,2)
    hazard_rate_01 (i) = {find(Hused(:,i) == 0.1)};
    hazard_rate_05 (i) = {find(Hused(:,i) == 0.5)};
    hazard_rate_09 (i) = {find(Hused(:,i) == 0.9)};
end

correct = choice_data == location_Rh_data;

%% Basic Plots

% Accuracy Calculations

for i = 1:size(change_data,2)
    percent_correct_low_haz(i) = sum(correct(hazard_rate_01{i},i))/size(hazard_rate_01{i},1);
    percent_correct_med_haz(i) = sum(correct(hazard_rate_05{i},i))/size(hazard_rate_05{i},1);
    percent_correct_high_haz(i) = sum(correct(hazard_rate_09{i},i))/size(hazard_rate_09{i},1);
    % Switched to only Threshold Uniform and Bayes Normal
    if i == 1
        accuracy_graph_output(i,:) = [percent_correct_low_haz(i), percent_correct_med_haz(i),percent_correct_high_haz(i) ];
    elseif i == 6
        accuracy_graph_output(2,:) = [percent_correct_low_haz(i), percent_correct_med_haz(i),percent_correct_high_haz(i) ];
    end
end

figure;

subplot(1,3,1)

% Plots for Accuracy
bar(accuracy_graph_output.*100);
xlabel('Computer Model');
ylabel('Accuracy');
ylim([0,100]);

%50 Percent Chance Line
chanceline = refline(0, 50);
chanceline.Color = 'k';
legend('Hazard = 0.1', 'Hazard = 0.5', 'Hazard = 0.9', 'Chance Line');
title('Accuracy at Different Hazard Rates');

% Proportion of Trials With Switches

for i = 1:size(change_data,2)
    percent_switch_low_haz(i) = sum(switch_choice(hazard_rate_01{i}(hazard_rate_01{i}~=1000), i))/size(hazard_rate_01{i}~=1000,1);
    percent_switch_med_haz(i) = sum(switch_choice(hazard_rate_05{i}(hazard_rate_05{i}~=1000), i))/size(hazard_rate_05{i}~=1000,1);
    percent_switch_high_haz(i) = sum(switch_choice(hazard_rate_09{i}(hazard_rate_09{i}~=1000), i))/size(hazard_rate_09{i}~=1000,1);
    % Switched to only Threshold Uniform and Bayes Normal
    if i == 1
        switch_graph_output(i,:) = [percent_switch_low_haz(i), percent_switch_med_haz(i),percent_switch_high_haz(i) ];
    elseif i == 6
        switch_graph_output(2,:) = [percent_switch_low_haz(i), percent_switch_med_haz(i),percent_switch_high_haz(i) ];
    end
end

subplot(1,3,2)

bar(switch_graph_output);
xlabel('Computer Model');
ylabel('Switching Frequency');
ylim([0,1]);
legend('Hazard = 0.1', 'Hazard = 0.5', 'Hazard = 0.9');
title('Probability of Switching');

% Proportion of Trials Switches When Correct

for i = 1:size(change_data,2)
    percent_switch_correct_low_haz(i) = sum(switch_choice(hazard_rate_01{i}(hazard_rate_01{i}~=1000),i) == correct(hazard_rate_01{i}(hazard_rate_01{i}~=1000), i))/size(correct(hazard_rate_01{i}(hazard_rate_01{i}~=1000), i),1);
    percent_switch_correct_med_haz(i) = sum(switch_choice(hazard_rate_05{i}(hazard_rate_05{i}~=1000),i) == correct(hazard_rate_05{i}(hazard_rate_05{i}~=1000), i))/size(correct(hazard_rate_05{i}(hazard_rate_05{i}~=1000), i),1);
    percent_switch_correct_high_haz(i) = sum(switch_choice(hazard_rate_09{i}(hazard_rate_09{i}~=1000),i) == correct(hazard_rate_09{i}(hazard_rate_09{i}~=1000), i))/size(correct(hazard_rate_09{i}(hazard_rate_09{i}~=1000), i),1);
    % Switched to only Threshold Uniform and Bayes Normal
    if i == 1
        switch_corr_graph_output(i,:) = [percent_switch_correct_low_haz(i), percent_switch_correct_med_haz(i), percent_switch_correct_high_haz(i)];
    elseif i == 6
        switch_corr_graph_output(2,:) = [percent_switch_correct_low_haz(i), percent_switch_correct_med_haz(i), percent_switch_correct_high_haz(i)];
    end
end

subplot(1,3,3)

bar(switch_corr_graph_output.*100);
xlabel('Computer Model');
ylabel('Switching Frequency When Correct %');
ylim([0,100]);
legend('Hazard = 0.1', 'Hazard = 0.5', 'Hazard = 0.9');
title('Probability of Switching If Answer is Correct');

%% Splits Data By Reward Value

% Across All Trials. Not a very useful graph since it doesn't take into
% account hazard rate.

% for i = 1:size(change_data,2)
%     % Rewards go from -9 to 9
%     for rew_value = 1:19
%     % Finds total number of rewards
%     % Finds number of switches at each reward level
%     % Calculates Proportion of Switches
%         total_rew = sum(find(reward(:,i) == rew_value-10)~=1000);
%         numswitches = sum(switch_choice(find(reward(:,i) == rew_value-10)~=1000,i) == 1);
%         rew_switch(i,rew_value) = numswitches/total_rew;
%     end
% end

% Splits Reward into Hazard Rate

p_sw_low = zeros(size(change_data,2), 19);
p_sw_med = zeros(size(change_data,2), 19);
p_sw_high = zeros(size(change_data,2), 19);

for i = 1:size(change_data,2)
     % Rewards go from -9 to 9
     for rew_value = 1:19
         % Calculates Values at Each Hazard Rate That Gives You Back a Reward Value
         values_w_reward_haz_01 = hazard_rate_01{i}(find(reward(hazard_rate_01{i}(hazard_rate_01{i}~=1000),i) == rew_value - 10));
         p_sw_low(i,rew_value) = sum(switch_choice(values_w_reward_haz_01,i))/size(values_w_reward_haz_01,1);
         values_w_reward_haz_05 = hazard_rate_05{i}(find(reward(hazard_rate_05{i}(hazard_rate_05{i}~=1000),i) == rew_value - 10));
         p_sw_med(i,rew_value) = sum(switch_choice(values_w_reward_haz_05,i))/size(values_w_reward_haz_05,1);
         values_w_reward_haz_09 = hazard_rate_09{i}(find(reward(hazard_rate_09{i}(hazard_rate_09{i}~=1000),i) == rew_value - 10));
         p_sw_high(i,rew_value) = sum(switch_choice(values_w_reward_haz_09,i))/size(values_w_reward_haz_09,1);
     end
end

figure;
%subplot(1,2,1)
%subplot(3,3,1);
p1 = plot(-9:9,p_sw_low(1,:),-9:9,p_sw_med(1,:),-9:9,p_sw_high(1,:));
p1(1).LineWidth = 2;
p1(2).LineWidth = 2;
p1(3).LineWidth = 2;
xlabel('Reward Value');
ylabel('Probability of Switching Next Trial');
ylim([0,1]);
title('Threshold Model');

% subplot(3,3,2);
% plot(-9:9,p_sw_low(2,:),-9:9,p_sw_med(2,:),-9:9,p_sw_high(2,:));
% xlabel('Reward Value');
% ylabel('Probability of Switching Next Trial');
% ylim([0,1]);
% title('Threshold Model - Normal');
% 
% subplot(3,3,3);
% plot(-9:9,p_sw_low(3,:),-9:9,p_sw_med(3,:),-9:9,p_sw_high(3,:));
% xlabel('Reward Value');
% ylabel('Probability of Switching Next Trial');
% ylim([0,1]);
% title('Threshold Model - Identical');
% 
% subplot(3,3,4);
% plot(-9:9,p_sw_low(4,:),-9:9,p_sw_med(4,:),-9:9,p_sw_high(4,:));
% xlabel('Reward Value');
% ylabel('Probability of Switching Next Trial');
% ylim([0,1]);
% title('Bayes Model - Uniform');
% 
% subplot(3,3,5);
% plot(-9:9,p_sw_low(5,:),-9:9,p_sw_med(5,:),-9:9,p_sw_high(5,:));
% xlabel('Reward Value');
% ylabel('Probability of Switching Next Trial');
% ylim([0,1]);
% title('Bayes Model - Normal');

%subplot(1,2,2);
figure;
p2 = plot(-9:9,p_sw_low(6,:),-9:9,p_sw_med(6,:),-9:9,p_sw_high(6,:));
p2(1).LineWidth = 2;
p2(2).LineWidth = 2;
p2(3).LineWidth = 2;
xlabel('Reward Value');
ylabel('Probability of Switching Next Trial');
ylim([0,1]);
title('Bayes Model');

% subplot(3,3,7);
% plot(-9:9,p_sw_low(7,:),-9:9,p_sw_med(7,:),-9:9,p_sw_high(7,:));
% xlabel('Reward Value');
% ylabel('Probability of Switching Next Trial');
% ylim([0,1]);
% title('DecCriterion Model - Uniform');
% 
% subplot(3,3,8);
% plot(-9:9,p_sw_low(8,:),-9:9,p_sw_med(8,:),-9:9,p_sw_high(8,:));
% xlabel('Reward Value');
% ylabel('Probability of Switching Next Trial');
% ylim([0,1]);
% title('DecCriterion Model - Normal');
% 
% subplot(3,3,9);
% plot(-9:9,p_sw_low(9,:),-9:9,p_sw_med(9,:),-9:9,p_sw_high(9,:));
% xlabel('Reward Value');
% ylabel('Probability of Switching Next Trial');
% ylim([0,1]);
% legend('Hazard = 0.1', 'Hazard = 0.5', 'Hazard = 0.9');
% title('DecCriterion Model - Identical');

%% Logs Odds Ratio for Reward High on Right

% Retrieve Reward Value For Current Trial
% Do NORMPDF with Reward Value on RH and RL with SIGMA_H and SIGMA_L

normpdf_high = normpdf(reward,RH_MU, SIGMA_H);
normpdf_low = normpdf(reward, RL_MU, SIGMA_L);
ratios = normpdf_high./normpdf_low;

% Calculates LLR
llr = log(ratios);

% Convert to Probability SPace
P = 1./(1+exp(-llr));

%P = 1./(1+exp(llr));

% We Need to Fix Values For Which the Choice was Left

for i = 1:size(change_data,2)
    % Finds position where choice was left
    leftchoice = find(choice_data(:,i) == 0);
    % P left is 1 - P Right
    P(leftchoice,i) = 1 - P(leftchoice,i);
    % P(leftchoice,i) = 1./(1+exp(-llr(leftchoice,i)));
end

% Derive LLR Again!

llr_right = -log(1./P - 1);

% Groups data into bins from - 15 to 15
bins = [-14:2:15];
% Manually Limited to Not 1000
binned_data = discretize(llr_right(1:999,1:9), bins);
for i = 1:size(change_data,2)
    for val = 1:size(bins,2)
        
        % Splits by Hazard Rate
        % Goes between 1 - 41 bins, finding average
        binval_01 = binned_data(hazard_rate_01{:,i}(hazard_rate_01{i}~=1000), i);
        indices_01 = hazard_rate_01{:,i}(hazard_rate_01{i}~=1000);
        options_for_val_01 = indices_01(find(binval_01  == val));
        prob_choosing_01(val,i) = sum(choice_data(options_for_val_01 + 1,i))./size(options_for_val_01,1);
        
        binval_05 = binned_data(hazard_rate_05{:,i}(hazard_rate_05{i}~=1000), i);
        indices_05 = hazard_rate_05{:,i}(hazard_rate_05{i}~=1000);
        options_for_val_05 = indices_05(find(binval_05  == val));
        prob_choosing_05(val,i) = sum(choice_data(options_for_val_05 + 1,i))./size(options_for_val_05,1);
        
        binval_09 = binned_data(hazard_rate_09{:,i}(hazard_rate_09{i}~=1000), i);
        indices_09 = hazard_rate_09{:,i}(hazard_rate_09{i}~=1000);
        options_for_val_09 = indices_09(find(binval_09  == val));
        prob_choosing_09(val,i) = sum(choice_data(options_for_val_09 + 1,i))./size(options_for_val_09,1);
        
    end
end

xaxis = -14:2:15;
figure;
%subplot(1,2,1);
p3 = plot(xaxis,prob_choosing_01(:,1), xaxis, prob_choosing_05(:,1), xaxis, prob_choosing_09(:,1));
p3(1).LineWidth = 2;
p3(2).LineWidth = 2;
p3(3).LineWidth = 2;
xlabel('Log Likelihood Ratio For High Reward on Right');
ylabel('Probability of Choosing Right Next Turn');
ylim([0,1]);
title('Threshold Model');

% subplot(3,3,2);
% plot(xaxis,prob_choosing_01(:,2), xaxis, prob_choosing_05(:,2), xaxis, prob_choosing_09(:,2));
% xlabel('Log Likelihood Ratio For High Reward on Right');
% ylabel('Probability of Choosing Right Next Turn');
% ylim([0,1]);
% title('Threshold Model - Normal');
% 
% subplot(3,3,3);
% plot(xaxis,prob_choosing_01(:,3), xaxis, prob_choosing_05(:,3), xaxis, prob_choosing_09(:,3));
% xlabel('Log Likelihood Ratio For High Reward on Right');
% ylabel('Probability of Choosing Right Next Turn');
% ylim([0,1]);
% title('Threshold Model - Identical');
% 
% subplot(3,3,4);
% plot(xaxis,prob_choosing_01(:,4), xaxis, prob_choosing_05(:,4), xaxis, prob_choosing_09(:,4));
% xlabel('Log Likelihood Ratio For High Reward on Right');
% ylabel('Probability of Choosing Right Next Turn');
% ylim([0,1]);
% title('Bayes Model - Uniform');
% 
% subplot(3,3,5);
% plot(xaxis,prob_choosing_01(:,5), xaxis, prob_choosing_05(:,5), xaxis, prob_choosing_09(:,5));
% xlabel('Log Likelihood Ratio For High Reward on Right');
% ylabel('Probability of Choosing Right Next Turn');
% ylim([0,1]);
% title('Bayes Model - Normal');
figure;
%subplot(1,2,2);
p4 = plot(xaxis,prob_choosing_01(:,6), xaxis, prob_choosing_05(:,6), xaxis, prob_choosing_09(:,6));
p4(1).LineWidth = 2;
p4(2).LineWidth = 2;
p4(3).LineWidth = 2;
xlabel('Log Likelihood Ratio For High Reward on Right');
ylabel('Probability of Choosing Right Next Turn');
ylim([0,1]);
title('Bayes Model');
% 
% subplot(3,3,7);
% plot(xaxis,prob_choosing_01(:,7), xaxis, prob_choosing_05(:,7), xaxis, prob_choosing_09(:,7));
% xlabel('Log Likelihood Ratio For High Reward on Right');
% ylabel('Probability of Choosing Right Next Turn');
% ylim([0,1]);
% title('DecCriterion Model - Uniform');
% 
% subplot(3,3,8);
% plot(xaxis,prob_choosing_01(:,8), xaxis, prob_choosing_05(:,8), xaxis, prob_choosing_09(:,8));
% xlabel('Log Likelihood Ratio For High Reward on Right');
% ylabel('Probability of Choosing Right Next Turn');
% ylim([0,1]);
% title('DecCriterion Model - Normal');
% 
% subplot(3,3,9);
% plot(xaxis,prob_choosing_01(:,9), xaxis, prob_choosing_05(:,9), xaxis, prob_choosing_09(:,9));
% xlabel('Log Likelihood Ratio For High Reward on Right');
% ylabel('Probability of Choosing Right Next Turn');
% ylim([0,1]);
%legend('Hazard = 0.1', 'Hazard = 0.5', 'Hazard = 0.9');
% title('DecCriterion Model - Identical');

%% Logit Fitting For Modeled Data

curvefit_threshnorm_01 = prob_choosing_01(:,1);
curvefit_threshnorm_05 = prob_choosing_05(:,1);
curvefit_threshnorm_09 = prob_choosing_09(:,1);

curvefit_bayesnorm_01 = prob_choosing_01(:,6);
curvefit_bayesnorm_05 = prob_choosing_05(:,6);
curvefit_bayesnorm_09 = prob_choosing_09(:,6);

rewrange = -9:9;

curvefit_rewthresh_01= p_sw_low(1,:);
curvefit_rewthresh_05 = p_sw_med(1,:);
curvefit_rewthresh_09 = p_sw_high(1,:);

curvefit_rewbayes_01= p_sw_low(6,:);
curvefit_rewbayes_05 = p_sw_med(6,:);
curvefit_rewbayes_09 = p_sw_high(6,:);

[results, gof] = createFits1(xaxis, curvefit_threshnorm_01, curvefit_threshnorm_05, curvefit_threshnorm_09, curvefit_bayesnorm_01, curvefit_bayesnorm_05, curvefit_bayesnorm_09, rewrange, curvefit_rewthresh_01, curvefit_rewthresh_05, curvefit_rewthresh_09, curvefit_rewbayes_01, curvefit_rewbayes_05, curvefit_rewbayes_09);

% % Plots the LLR Fits

figure;
pl1 = plot(results{1});
pl1.Color = [0 0.4470 0.7410];
pl1.LineWidth = 2;
hold;
pl2 = plot(results{2});
pl2.LineWidth = 2;
pl2.Color = [0.8500 0.3250 0.0980];
pl3 = plot(results{3});
pl3.Color = [0.9290 0.6940 0.1250];
pl3.LineWidth = 2;
hold off;
xlabel('Log Likelihood Ratio For High Reward on Right')'
ylabel('Probability of Choosing Right Next Turn');
title('Curve Fits for Threshold Model - LLR');
%legend('10% Hazard Rate', '50% Hazard Rate', '90% Hazard Rate');
 
figure;
pl4 = plot(results{4});
pl4.Color = [0 0.4470 0.7410];
pl4.LineWidth = 2;
hold;
pl5 = plot(results{5});
pl5.LineWidth = 2;
pl5.Color = [0.8500 0.3250 0.0980];
pl6 = plot(results{6});
pl6.Color = [0.9290 0.6940 0.1250];
pl6.LineWidth = 2;
hold off;
xlabel('Log Likelihood Ratio For High Reward on Right')'
ylabel('Probability of Choosing Right Next Turn');
title('Curve Fits for Bayes Model - LLR');
%legend('10% Hazard Rate', '50% Hazard Rate', '90% Hazard Rate');
 
figure;
pl7 = plot(results{7});
pl7.Color = [0 0.4470 0.7410];
pl7.LineWidth = 2;
hold;
pl8 = plot(results{8});
pl8.LineWidth = 2;
pl8.Color = [0.8500 0.3250 0.0980];
pl9 = plot(results{9});
pl9.Color = [0.9290 0.6940 0.1250];
pl9.LineWidth = 2;
hold off;
xlabel('Reward Gained at Current Trial')
ylabel('Probability of Switching Next Turn');
title('Curve Fits for Threshold - Reward');
%legend('10% Hazard Rate', '50% Hazard Rate', '90% Hazard Rate');


figure;
pl10 = plot(results{10});
pl10.Color = [0 0.4470 0.7410];
pl10.LineWidth = 2;
hold;
pl11 = plot(results{11});
pl11.LineWidth = 2;
pl11.Color = [0.8500 0.3250 0.0980];
pl12 = plot(results{12});
pl12.Color = [0.9290 0.6940 0.1250];
pl12.LineWidth = 2;
hold off;
xlabel('Reward Gained at Current Trial')'
ylabel('Probability of Switching Next Turn');
title('Curve Fits for Bayes - Reward');
%legend('10% Hazard Rate', '50% Hazard Rate', '90% Hazard Rate');

%% Percent Correct in 4 Trials Immediately Following Switch

% Finds trials where the hazard rate is constant but switch in reward poisiton occurred
% Trial before switch is at same hazard rate

% Indicate Change Happen AT THE NEXT INDEX, not the CURRENT one where it
% says there is a difference. This is important for the plots moving forward.
switches = abs(location_Rh_data(2:1000,:) - location_Rh_data(1:999,:));

% Adds 1 to indicate that it's actually 1 point over. For hazard rate 01,
% make sure that the next step after is not a switch since it's supposed to
% be constant after.
for i = 1:size(change_data,2)
    hazard_01_switch (i) = {find(Hused(1:994,i) == 0.1 & Hused(7:1000,i) == 0.1 & switches(1:994,i) == 1 & switches(2:995, i) == 0) + 1};
    hazard_05_switch (i) = {find(Hused(1:994,i) == 0.5 & Hused(7:1000,i) == 0.5 & switches(1:994,i) == 1) + 1};
    hazard_09_switch (i) = {find(Hused(1:994,i) == 0.9 & Hused(7:1000,i) == 0.9 & switches(1:994,i) == 1) + 1};
end

for i = 1:size(change_data,2)
    
    % Accuracy Three - A Few Trials Following
    indices_st01 = hazard_01_switch{i};
    switch_correct_01 (i) = sum(correct(indices_st01 + 3) + correct(indices_st01 + 4) + correct(indices_st01 + 5) + correct(indices_st01 + 6))./size(indices_st01,1)./4.*100;
    indices_st05 = hazard_05_switch{i};
    switch_correct_05 (i) = sum(correct(indices_st05 + 3) + correct(indices_st05 + 4) + correct(indices_st05 + 5) + correct(indices_st05 + 6))./size(indices_st05,1)./4.*100;
    indices_st09 = hazard_09_switch{i};
    switch_correct_09 (i) = sum(correct(indices_st09 + 3) + correct(indices_st09 + 4) + correct(indices_st09 + 5) + correct(indices_st09 + 6))./size(indices_st09,1)./4.*100;
    
    % Puts Following into a Graph
    graph_output_imm (i,:) = [switch_correct_01(i), switch_correct_05(i), switch_correct_09(i)];
        
    % Accuracy At Change Point
    switch_pt_accuracy(i,:) = [mean(correct(indices_st01)).*100, mean(correct(indices_st05)).*100, mean(correct(indices_st09)).*100];
    
    % No Switch Accuracy
    no_switch_pt_accuracy(i,:) = [mean(correct(~ismember(find(Hused(1:994,i) == 0.1 & Hused(7:1000,i) == 0.1),indices_st01))).*100, mean(correct(~ismember(find(Hused(1:994,i) == 0.5 & Hused(7:1000,i) == 0.5),indices_st05))).*100, mean(correct(~ismember(find(Hused(1:994,i) == 0.9 & Hused(7:1000,i) == 0.9),indices_st09))).*100];

    % Prob Switching After Change Point
    % 5 Immediate Steps Afterwards
     prob_switch_01 (i) = sum(switch_choice(indices_st01 + 1) + switch_choice(indices_st01 + 2) + switch_choice(indices_st01 + 3) + switch_choice(indices_st01 + 4) + switch_choice(indices_st01 + 5))./size(indices_st01,1)./5.*100;
     prob_switch_05 (i) = sum(switch_choice(indices_st05 + 1) + switch_choice(indices_st05 + 2) + switch_choice(indices_st05 + 3) + switch_choice(indices_st05 + 4) + switch_choice(indices_st05 + 5))./size(indices_st05,1)./5.*100;
     prob_switch_09 (i) = sum(switch_choice(indices_st09 + 1) + switch_choice(indices_st09 + 2) + switch_choice(indices_st09 + 3) + switch_choice(indices_st09 + 4) + switch_choice(indices_st09 + 5))./size(indices_st09,1)./5.*100;

end

figure;
subplot(1,3,1);
bar(switch_pt_accuracy);
chanceline = refline(0, 50);
chanceline.Color = 'k';
xlabel('Computer Model');
ylabel('Percent Correct At Change Point');
ylim([0,100]);
legend('Hazard = 0.1', 'Hazard = 0.5', 'Hazard = 0.9', 'Chance Line');
title('% Correct At Change Point');

subplot(1,3,2);
bar(graph_output_imm);
chanceline = refline(0, 50);
chanceline.Color = 'k';
xlabel('Computer Model');
ylabel('Percent Correct');
ylim([0,100]);
legend('Hazard = 0.1', 'Hazard = 0.5', 'Hazard = 0.9', 'Chance Line');
title('% Correct in the Trials Following Switch Across Hazard Rates');

subplot(1,3,3);
bar(graph_output_imm - switch_pt_accuracy);
xlabel('Computer Model');
ylabel('Percent Improvement');
ylim([-20,20]);
legend('Hazard = 0.1', 'Hazard = 0.5', 'Hazard = 0.9');
title('% Average Improvement');

% Percent Correct in Trials Without Switch

figure;
subplot(1,2,1)
bar(no_switch_pt_accuracy);
chanceline = refline(0, 50);
chanceline.Color = 'k';
xlabel('Computer Model');
ylabel('Percent Correct');
ylim([0,100]);
legend('Hazard = 0.1', 'Hazard = 0.5', 'Hazard = 0.9', 'Chance Line');
title('% Correct When No Change Point');

subplot(1,2,2);
bar(no_switch_pt_accuracy - switch_pt_accuracy);
xlabel('Computer Model');
ylabel('Percent Difference in Performance');
ylim([-20,20]);
legend('Hazard = 0.1', 'Hazard = 0.5', 'Hazard = 0.9');
title('% Difference Between No Switch and Switch');

figure;
bar([prob_switch_01; prob_switch_05; prob_switch_09]');
xlabel('Computer Model');
ylabel('Percent Likelihood of Switching');
ylim([0,100]);
legend('Hazard = 0.1', 'Hazard = 0.5', 'Hazard = 0.9');
title('% Likelihood of Switching After Change Point');

%% Probability Correct After Hazard Block Change

switches_haz = abs(Hused(2:996,:) - Hused(1:995,:));

for i = 1:size(change_data,2)
    hazard_01_05_switch (i) = {find(Hused(1:995,i) == 0.1 & Hused(2:996, i) == 0.5)};
    hazard_05_09_switch (i) = {find(Hused(1:995,i) == 0.5 & Hused(2:996, i) == 0.9)};
    hazard_01_09_switch (i) = {find(Hused(1:995,i) == 0.1 & Hused(2:996, i) == 0.9)};
    hazard_05_01_switch (i) = {find(Hused(1:995,i) == 0.5 & Hused(2:996, i) == 0.1)};
    hazard_09_05_switch (i) = {find(Hused(1:995,i) == 0.9 & Hused(2:996, i) == 0.5)};
    hazard_09_01_switch (i) = {find(Hused(1:995,i) == 0.9 & Hused(2:996, i) == 0.1)};
    all_switch (i) = {find(switches_haz(:,i) ~= 0) + 1};
end

for i = 1:size(change_data,2)
    indices_st0105 = hazard_01_05_switch{i};
    if size (indices_st0105,1) > 0
        switch_correct_01_05 (i) = sum(correct(indices_st0105 + 1) + correct(indices_st0105 + 2) + correct(indices_st0105 + 3) + correct(indices_st0105 + 4))./size(indices_st0105,1)./4;
    else
        switch_correct_01_05 (i) = NaN;
    end
    
    indices_st0509 = hazard_05_09_switch{i};
    if size (indices_st0509,1) > 0
        switch_correct_05_09 (i) = sum(correct(indices_st0509 + 1) + correct(indices_st0509 + 2) + correct(indices_st0509 + 3) + correct(indices_st0509 + 4))./size(indices_st0509,1)./4;
    else
        switch_correct_05_09 (i) = NaN;
    end
    
    indices_st0109 = hazard_01_09_switch{i};
    if size (indices_st0109,1) > 0
        switch_correct_01_09 (i) = sum(correct(indices_st0109 + 1) + correct(indices_st0109 + 2) + correct(indices_st0109 + 3) + correct(indices_st0109 + 4))./size(indices_st0109,1)./4;
    else
        switch_correct_01_09 (i) = NaN;
    end
    
    indices_st0501 = hazard_05_01_switch{i};
    if size (indices_st0501,1) > 0
        switch_correct_05_01 (i) = sum(correct(indices_st0501 + 1) + correct(indices_st0501 + 2) + correct(indices_st0501 + 3) + correct(indices_st0501 + 4))./size(indices_st0501,1)./4;
    else
        switch_correct_05_01 (i) = NaN;
    end
    
    indices_st0905 = hazard_09_05_switch{i};
    if size (indices_st0905,1) > 0
        switch_correct_09_05 (i) = sum(correct(indices_st0905 + 1) + correct(indices_st0905 + 2) + correct(indices_st0905 + 3) + correct(indices_st0905 + 4))./size(indices_st0905,1)./4;
    else
        switch_correct_09_05 (i) = NaN;
    end
    
    indices_st0901 = hazard_09_01_switch{i};
    if size (indices_st0901,1) > 0
        switch_correct_09_01 (i) = sum(correct(indices_st0901 + 1) + correct(indices_st0901 + 2) + correct(indices_st0901 + 3) + correct(indices_st0901 + 4))./size(indices_st0901,1)./4;
    else
        switch_correct_09_01 (i) = NaN;
    end
    
    indices_all = all_switch{i};
    switch_all (i) = sum(correct(indices_all + 1) + correct(indices_all + 2) + correct(indices_all + 3) + correct(indices_all + 4))./size(indices_all,1)./4;

end

figure;
subplot(1,3,1)
bar([switch_correct_01_05; switch_correct_01_09]');
xlabel('Computer Model');
ylabel('Accuracy');
ylim([0,1]);
legend('To Hazard = 0.5', 'To Hazard = 0.9');
title('Accuracy When Switching From Blocks of Hazard Rate = 0.1');

subplot(1,3,2)
bar([switch_correct_05_09; switch_correct_05_01]');
xlabel('Computer Model');
ylabel('Accuracy');
ylim([0,1]);
legend('To Hazard = 0.9', 'To Hazard = 0.1');
title('Accuracy When Switching From Blocks of Hazard Rate = 0.5');

subplot(1,3,3)
bar([switch_correct_09_01; switch_correct_09_05]');
xlabel('Computer Model');
ylabel('Accuracy');
ylim([0,1]);
legend('To Hazard = 0.1', 'To Hazard = 0.5');
title('Accuracy When Switching From Blocks of Hazard Rate = 0.9');

figure;
bar(switch_all);
xlabel('Computer Model');
ylabel('Accuracy');
ylim([0,1]);
title('Overall Accuracy When Switching Between Blocks of Hazard Rates');

%% Reward Distribution

x = -10:0.1:10;
yh = normpdf(x,RH_MU, SIGMA_H);
yl = normpdf(x,RL_MU, SIGMA_L);
plot(x,yh,x,yl);
ylim([0,0.21]);
xlabel('Reward Value');
ylabel('Probability of Getting Reward');
legend('High Reward', 'Low Reward');
title('Reward Probability Distributions');