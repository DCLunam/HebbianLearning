%Hebian learning wit LTP and LTD
%Script used and described in 
%Tovar, A. E., Westermann, G., and Torres, A. (2018). 
%From altered synaptic plasticity to atypical learning: 
%a computational model of Down syndrome. Cognition, 171, 15-24. 
%https://doi.org/10.1016/j.cognition.2017.10.021
%Please cite this work if you use this script

%It uses Hebbian learning, and uses strenghtening and weakening of
%connections to stabilize and limit weight values. Connection weights are
%modeled following notions of long term potentiation and long term depression
%more info and comments Angel Tovar aetovar@unam.mx ;
%eugenio.tovar@gmail.com

function [W , W_total_epochs] = hebb_ltd(A , epochs , sequence_training ,  beta ,  thres)

%A = matrix os stimulation as trials(rows) and stimuli(columns) e.g., for AB, BC training [1 1 0;0 1 1]
%epochs = number of epochs
%sequence_training = 1 for random, 2 for sequencial
%beta = learning rate
%thres = threshold for LTP/LTD function

[trials, neurons]= size(A);
W = zeros(neurons);
beta = beta.*2;% this is needed to add noise and keep beta(mean) = beta input
    
for i = 1:epochs
    
    if sequence_training == 1
        A = A(randperm(size(A,1)),:); 
    else
        A = A;
    end
    

    for t = 1:trials
        %updating activations
        %first external activation (bottom up act)
        ext_act = A(t,:);
    
        %then internal activation (spread activation, top down act)
        act = repmat(ext_act,neurons,1); 
        int_act = dot(W,act');%Net input
        int_act_sigm = 1./(1+exp(-int_act)); % activation function, sigmoid
        int_act_fin = (int_act > 0.85) .* int_act_sigm; % activation function with threshold
       
        final_act = ext_act + int_act_fin;
        final_act = ((final_act<1).*final_act)+(final_act>=1);%limiting act values to avoid surpassing 1
    
        %Coactivations
        act_1 = repmat(final_act,neurons,1);
        act_2 = repmat(final_act',1,neurons);
        coactivation = (act_1 .*act_2);

        %learning
        LamdaLTP = ((coactivation > thres) .* coactivation) - (W);
            for la=1:neurons
                LamdaLTP(la,la)  = 0;
            end
                
        alpha_positiveP = (LamdaLTP>0).*(LamdaLTP.*(beta.*rand)); 
        alpha_negativeP = (LamdaLTP<0).*(LamdaLTP.*(beta.*rand)); 
        alpha = alpha_positiveP  +  alpha_negativeP;    

        delta = (act_1.*act_2).*alpha;
        W = W+delta;
        W_total_epochs(:,:,i) = W;
    end
end



