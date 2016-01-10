for t = 1:m
    %For the input layer, where l=1:
    a1 = [1; X(t,:)'];

    %For the hidden layers, where l=2:
    z2 = Theta1 * a1';
    a2 = [1; sigmoid(z2)];

    %For the hidden layers, where l=3:
    z3 = Theta2 * a2;
    a3 = [1; sigmoid(z3)];

    delta_3 = (Theta3' * delta_4) .* [1; sigmoidGradient(z3)];
    delta_3 = delta_3(2:end); %Taking of the bias row

    delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)];
    delta_2 = delta_2(2:end); %Taking of the bias row

    %delta_1 is not calculated because we do not associate error with the input    

    %Big delta update
    Theta1_grad = Theta1_grad + delta_2 * a1';
    Theta2_grad = Theta2_grad + delta_3 * a2';
    

 end

%This is not part of the for-loop:
 Theta1_grad = (1/m) * Theta1_grad ;
 Theta2_grad = (1/m) * Theta2_grad;

