%Angel Tovar - aetovar@unam.mx , eugenio.tovar@gmail.com
%Last update: April 2022

%The learning algorithm is a Hebbian rule that implements strengthening
%and weakening of connections to stabilize and limit weight values.
%Weight changes are implemented following notions from long-term
%potentiation, long-term depression, and metaplasticity.
%See the full rationale and theoretical background of the learning
%algorithm in: 
%Tovar, Á. E., & Westermann, G. (2017). A Neurocomputational Approach to
%Trained and Transitive Relations in Equivalence Classes.
%Frontiers in Psychology, 8. https://doi.org/10.3389/fpsyg.2017.01848
%
%Tovar, Á. E., Westermann, G., & Torres, A. (2018). From altered synaptic
%plasticity to atypical learning: A computational model of Down syndrome.
%Cognition, 171, 15-24. https://doi.org/10.1016/j.cognition.2017.10.021

function [W] = TW_hebb_ltd(Matrix,beta,ltp_thres,epochs)
%% Input Arguments %
%Matrix - Input to the model. Must have object (syllables, words, visual objects, etc) codifications in columns for each time step in rows.
%beta   - learning rate, usually 0.2
%ltp_thres - Threshold for negative/positive (LTD/LTP) weight changes. usually between 0.6 and 0.7 for typical development. Adjust values if other activation functions are used 
%epochs  - number of repetitions of the input Matrix during training/familiarization

%% Outputs
%W - Weight Matrix
%Other outputs can be included such as W_total_epochs

%% Initialize Weight Matrix
[trials,neurons] = size(Matrix);
W = zeros(neurons);%could also be initialized with small random values (e.g., W = rand(neurons)*.1)
%beta = input arg, set to 0.2 or 0.3, etc.
%beta2 = (beta.*0.1), or (beta.*-0.1); use a different value for weight decays when needed

%% Familiarization
for i = 1:epochs %epcohs loop
%%
    for t = 2:trials %% training trials loop 
        %% updating activations
        %first external activation (and activation decay from previous time step)
        ext_act = Matrix(t,:) + (Matrix(t-1,:).*.9);

        %then internal activation (spreading activation)
        act = repmat(ext_act,neurons,1); 
        int_act = dot(W,act');%Net input
        int_act_sigm = 1./(1+exp(-int_act)); % activation function, sigmoid
        int_act_fin = (int_act > 0.5) .* int_act_sigm; % activation function with threshold. The threshold is used to transform relevant input into activation values, otherwise 0 and negative values are also transformed into positive activations
        %Alternative activation function:
        %int_act_fin =  int_act ./ (1+int_act);% This one also works, but LTP threshold must be lower. For example, Typical LTP threshold values = [0.3 , 0.35] 

        final_act = ext_act + int_act_fin; %External and spreading activations
        final_act = ((final_act<1).*final_act)+(final_act>=1);%limiting act values in a 0 - 1  range
        
        %Coactivation matrix
        act_1 = repmat(final_act,neurons,1);
        act_2 = repmat(final_act',1,neurons);
        coactivation = (act_1 .*act_2);

        %% Hebbian learning
        %LTP/LTD Threshold (lambda)
        LambdaLTP = ((coactivation > ltp_thres) .* coactivation) - (W);
        
        for la=1:neurons
            LambdaLTP(la,la)  = 0;%sets diagonal to 0 when self connections are not modeled. Otherwise comment these lines to use self connections
        end

        beta_positive = (LambdaLTP>=0).*(LambdaLTP.*(beta .*rand)); %rand adds noise and stochasticity to the model.
        beta_negative = (LambdaLTP<0).*(LambdaLTP.*(beta .*rand)); %Change beta term to beta2 when beta2 values are used, as in Tovar and Westermann (2017), otherwise use beta 
        beta_ltp = beta_positive  +  beta_negative;    

        learning_delta = (act_1.*act_2).*beta_ltp;%Hebbian learning
        W = W+learning_delta;%updating W
        %W_trials(:,:,t) = W; 
    end
    W_total_epochs(:,:,i) = W;%Saving all Ws
end
