function [y] = rastrigin(x)
% REFERENCE: 
% Yao, Xin, Yong Liu, and Guangming Lin. "Evolutionary programming made 
% faster." IEEE Transactions on Evolutionary computation 3.2 (1999): 82-102.
%
% Author: Jialin Liu
% email: jialin.liu.cn@gmail.com
% Website: http://www.liujialin.tech/
% Feb 2019; Last revision: 19-Feb-2019
if length(x)~=30
    error('Error. \nInput must be of dimension 30. The dimension of current input is %d.', length(x));
end
if min(x)<-5.12
    error('Error. \n At least one element exceeds the lower bound.')
end
if max(x)>5.12
    error('Error. \n At least one element exceeds the upper bound.')
end
y=sum(x.^2-10*cos(2*pi*x)+10);
end