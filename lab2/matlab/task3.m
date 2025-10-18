%% Задание 3
% Исходные данные
A = [0 1
    2 0];
B = [0; 1];
x0 = [1; 1];
a = 1.99999;

% LMI
cvx_begin sdp
variable P(2,2)
variable Y(1,2)
variable mumu
minimize mumu
P > 0.0001*eye(2);
P*A' + A*P + 2*a*P + Y'*B' + B*Y <= 0;
[P x0;
 x0' 1] > 0;
[P Y';
 Y mumu] > 0;
cvx_end

% disp("Без ограничения управления");
eig(A)
P
K = Y*P^-1
eig(A+B*K)

% % LMI с минимизацией
% cvx_begin sdp
% variable P2(2,2)
% variable Y2(1,2)
% variable mumu
% minimize mumu
% P2 > 0.0001*eye(2);
% P2*A' + A*P2 + 2*a*P2 + Y2'*B' + B*Y2 <= 0;
% [P2 x0;
%  x0' 1] > 0;
% [P2 Y2';
%  Y2 mumu] > 0;
% cvx_end
