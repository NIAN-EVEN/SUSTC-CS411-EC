function [apprx,appry]=optimise(funcName,budget,crossoverIdx,mutationIdx,selectionIdx)
warning on MATLAB:divideByZero
% Input:
% crossoverIdx: the index of crossover operator \in {1, 2, 3}
% mutationIdx: the index of mutation operator \in {1, 2, 3}
% selectionIdx: the index of selection operator \in {1, 2, 3}
%
% Output:
% apprx: the approximate optimum
% appry: the approximate optimal value
%
%% TODO: below to implement your own EA
%% You may need to provide multiple files if you are asked to implement 
%% more than one algorithm or an algorithm with different configurations.
%------------- BEGIN CODE --------------
% get configureation: 
% chromesome length
% upper and lower bound of each gene
dimension=config(1);
lower_bounc=config(2);
upper_bound=config(3);
pop_size=100;
% init population
% determine each chrosome's gene
% calculate each chrosome's score
% rank the population
pop.vec=rand(dimension,pop_size)*(upper_bound-lower_bound)+lower_bound;
pop.yita=rand(dimension,pop_size)
% select parents


% crossover to get offspring: expand good gene

% mutation to expand search area

% select chrosome to survive


%------------- END OF CODE --------------
end

function y=fitness(funcName,x)
eval(sprintf('objective=@%s;',funcName)); % Do not delete this line
%% TODO: below to implement your own fitness function
%------------- BEGIN CODE --------------
objValue=objective(x); 
y=objValue;
%------------- END OF CODE --------------
end

% class chromesome
% 1 for multimodel
% 1 for unimodel
% 1 for step function

% 3 x crossover function: combination of good properties
% identify which dim better
% 交叉的概率取决于向某个方向发展取得的成长/步长
% 1. discrete
% 2. arithmetic
% 3. intermidiate
% 4. quadratic
% 5 .heuristic
% 3 x mutation function: broaden search space, search diversity
% 1. IFEP
function [x_prime,yita_prime] = IFEP_mutation(x, yita)
gaussian_random = normrnd(0,1,length(yita),1)
% To generate N random values of x with a Cauchy distribution 
% where b is the half width at the half maximum density level and m is the statistical median:
% x = m+b*tan(pi*(rand(N,1)-1/2));
% cited from: https://ww2.mathworks.cn/matlabcentral/answers/335701-how-to-draw-random-number-from-a-cauchy-distribution
cauchy_random_j = tan(pi*(rand(len(x),1)-0.5))
tao_prime=1/sqrt(2*length(x))
tao=1/sqrt(2*sqrt(length(x)))

x_prime=x+yita.*gaussian_random
yita_prime=yita.*exp(tao_prime*gaussian_random+tao*cauchy_random_j)
end
% 2. CEP
% 3. FEP
% 3 x selection function: 
