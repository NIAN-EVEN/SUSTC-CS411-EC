clear all;
global initial_flag; % the global flag used in test suite 

func_test = [1 6 11 13]; 
func_c(1)=2; func_c(6)=0.1; func_c(11)=0.05; func_c(13)=0.5;
func_r(1)=0.01; func_r(6)=0.5; func_r(11)=0.01; func_r(13)=0.01;
accs = [1 0.1 0.01 0.001 0.0001 0.00001];
algo_n(1)=400; algo_n(2)=100; algo_n(3)=400; 

algorithm={@GA};
% algorithm={@GA,@DE,@SAES};

repeat = 10;

for i=1:length(func_test)
    for a=1:length(algorithm)
        algo = algorithm{a};
        func_num = func_test(i);
        c = func_c(func_num);
        accuracy = 0.1;
        dim = get_dimension(func_num);
        budget = get_maxfes(func_num);
        lb = get_lb(func_num);lb=lb(1);
        ub = get_ub(func_num);ub=ub(1);
        opt = get_fgoptima(func_num);
        optnum = get_no_goptima(func_num);
        n = algo_n(a);
        s = zeros(1,6);
        for k=1:repeat
            initial_flag = 0; 
            [population,appry,apprx] = algo(func_num, budget, dim, lb, ub, n, 0.00001);
            for j=1:n
                if population{j,1}+0.00001 < appry
                    population(j,:) = {appry, apprx, appry};
                    break;
                end
            end
%             disp(population);
            population = cell2mat(population(:,2));
            
            for j=1:6
                initial_flag = 0; 
                acc = accs(j);
                [count, goptima_found] = count_goptima(population, func_num, acc);
%                  fprintf('f_%02d, opt:%.3f, appry:%.3f, optnum: %d found:%d!\n', func_num,opt,appry,optnum,count);
                s(j) = s(j) + count;
            end

        % 	if count ~=0
        % 		goptima_found;
        % 		for i=1:size(goptima_found,1)
        % 			val = niching_func(goptima_found(i,:), func_num);
        % 			fprintf('F_p: %f, F_g:%f, diff: %f\n', val, get_fgoptima(func_num), abs(val - get_fgoptima(func_num)))
        % 			fprintf('F_p - F_g <= %f : %d\n', accuracy, abs(val - get_fgoptima(func_num))<accuracy )
        % 		end
        % 	end
        end
        fprintf('algo: %d, f_%02d, optnum: %02d, 0:%.2f, 1: %.2f, 2: %.2f, 3: %.2f, 4: %.2f, 5: %.2f \n', a, func_num, optnum, s(1)/repeat, s(2)/repeat, s(3)/repeat, s(4)/repeat, s(5)/repeat, s(6)/repeat);
    end
    fprintf('\n');
end
function [population,appry,apprx]=SAES(func_num, budget, dim, lb, ub, n, a)
    global initial_flag
    appry = -inf;
    lamda = n;
    population = init_population(n, func_num, dim, lb, ub);
    thegema = ones(n, 1)*(ub-lb)*0.1;
%     population = share_fitness(population, a, b, c);
    while budget > n
        offspring = cell(lamda, 3);
        %随机选择亲代杂交生成子代
        for k = 1:lamda
            thegema_ = thegema(k)*exp(normrnd(0,1));
            offspring{k, 2} = population{k, 2} + thegema_*normrnd(0, 1, 1, dim);
            offspring{k, 2} = fixbound(offspring{k, 2}, dim, lb, ub);
            offspring{k, 1} = fitness(offspring{k, 2}, func_num);
            budget = budget - 1;
            if offspring{k, 1} > population{k, 1}
                population(k, :) = offspring(k, :);
                thegema(k) = thegema_;
            end
            
        end

         %更新全局最优解,最差解
        [v, idx] = max(cell2mat(population(:,1)));
        if v > appry
            appry = v;
            apprx = population{idx,2};
        end
        
%         initial_flag = 0; 
%         [count, goptima_found] = count_goptima(cell2mat(population(:,2)), func_num, a);
%         fprintf("%d\n", count);
    end
end

function [population,appry,apprx]=GA(func_num, budget, dim, lb, ub, n, a)
    global initial_flag
    appry = -inf;
    F = 0.25;
    mutation_rate = zeros(n, 1);
    population = init_population(n, func_num, dim, lb, ub);
    offspring = cell(2, 3);
    for bu = n:n:budget
        %每2个个体为一组，协同进化
        for k = 1:2:n
            pidx = k+1;
            if population{k,1} > population{pidx,1}
                p1 = population{k,2};
                p2 = population{pidx,2};
            else
                p2 = population{k,2};
                p1 = population{pidx,2};
            end
            dif = p1-p2;
            c1 = p1 + F*dif;
            c2 = p2 + F*dif;
            offspring{1, 2} = c1;
            offspring{2, 2} = c2;

            for i = 1:2
                for j = 1:dim
                    if rand()<mutation_rate(k)
                        offspring{i, 2}(j) = offspring{i, 2}(j) + norm(dif)*normrnd(0, 1, 1, 1);
                    end
                end
                offspring{i, 2} = fixbound(offspring{i, 2}, dim, lb, ub);
                offspring{i, 1} = fitness(offspring{i, 2}, func_num);
            end
            
            flag = 0;
            if norm(population{k,2}-c1)+norm(population{pidx,2}-c2) <= norm(population{k,2}-c2)+norm(population{pidx,2}-c1)
               if offspring{1,1} >  population{k,1}
                   population(k,:) = offspring(1,:);
                   flag = 1;
               end
               if offspring{2,1} > population{pidx,1}
                   population(pidx,:) = offspring(2,:);
                   flag = 1;
               end
            else
               if offspring{2,1} > population{k,1}
                   population(k,:) = offspring(2,:);
                   flag = 1;
               end
               if offspring{1,1} > population{pidx,1}
                   population(pidx,:) = offspring(1,:);
                   flag = 1;
               end
            end
            if flag == 0
                mutation_rate(k) = 1;
            else
                mutation_rate(k) = 0;
            end
        end
        
         %更新全局最优解,最差解
        [v, idx] = max(cell2mat(population(:,1)));
        if v > appry
            appry = v;
            apprx = population{idx,2};
        end
        
%         initial_flag = 0; 
%         [count, goptima_found] = count_goptima(cell2mat(population(:,2)), func_num, a);
%         fprintf("%d\n", count);
    end
end

function [population,appry,apprx]=DE(func_num, budget, dim, lb, ub, n, a)
    global initial_flag
    appry = -inf;
    cross_rate = 0.5;
    F = 0.5;
    population = init_population(n, func_num, dim, lb, ub);
    for b = n:n:budget
        offspring = cell(n,3);
        dist = distmat(population);
        for i = 1:n
            rand_idx = randsample(n,4);
            R = randsample(dim,1);
            nn = nearest(dist, i);
            offspring{i,2}=population{nn,2}+F*(population{rand_idx(1),2}-population{rand_idx(2),2})+F*(population{rand_idx(3),2}-population{rand_idx(4),2});
            for j = 1:dim
                if j~=R && rand()<cross_rate
                    offspring{i, 2}(j) = population{i, 2}(j);
                end
            end
            offspring{i, 2} = fixbound(offspring{i, 2}, dim, lb, ub);
            offspring{i, 1} = fitness(offspring{i, 2}, func_num);
            if offspring{i, 1} > population{i, 1} + a
                population(i, :) = offspring(i, :);
            end
        end

        %更新全局最优解,最差解
        [v, idx] = max(cell2mat(offspring(:,1)));
        if v > appry
            appry = v;
            apprx = offspring{idx,2};
        end
        
%         initial_flag = 0; 
%         [count, goptima_found] = count_goptima(cell2mat(population(:,2)), func_num, a);
%         fprintf("%d\n", count);
    end
end

function child=arithmetic_recombination(parent)
    a = rand();
    child(1,:) = a*parent(1,:) + (1-a)*parent(2,:);
    child(2,:) = a*parent(2,:) + (1-a)*parent(1,:);
end

function population_idx=trunc_selection(population, n)
    [~,idx] = sort(cell2mat(population(:,3)),'descend');
    population_idx = idx(1:n,:);
%     population = population(idx,:);
%     population = population(1:n,:);
end

function population_idx=roulettewheel_selection(population, n)
    alpha = 5;
    worst = min(cell2mat(population(:,3)));
    best = max(cell2mat(population(:,3)));
    total = 0;
    new_value = zeros(1, length(population));
    for i=1:length(population)
        new_value(i) = exp(alpha*(population{i, 3} - worst)/(best-worst+0.001));
        total = total + new_value(i);
    end
    population_idx = zeros(n,1);
    for i=1:n
        pick = rand()*total;
        for j=1:length(population)
            pick = pick - new_value(j);
            if pick <= 0
                population_idx(i) = j;
                break;
            end
        end
    end
end

function offspring=fixbound(offspring, dim, lb, ub)
    for j = 1:dim
        if offspring(j) < lb
            offspring(j) = lb;
        end
        if offspring(j) > ub
            offspring(j) = ub;
        end
    end
end

function y=fitness(x, func_num)
    y = niching_func(x, func_num);
end

function pidx=nearest(dist, x)
    d = inf;
    for i=1:length(dist)
        if(dist(i,x)<d) && i~=x
            d=dist(i,x);
            pidx=i;
        end
    end
end

function dist=distmat(population)
    n = length(population);
    dist = zeros(n, n);
    for i=1:n
        for j=1:n
            if j<=i
                dist(i, j) = dist(j, i);
            else
                dist(i, j) = norm(population{i,2} - population{j,2});     
            end
        end
    end
end

function population=init_population(n, func_num, dim, lb, ub)
    population = cell(n, 3);
    for i=1:n
        population{i,2} = randrange(dim, lb, ub);
        population{i,1} = fitness(population{i,2}, func_num);
    end
end

function y=randrange(dim, lower, upper)
    y=rand(1,dim)*(upper-lower)+lower;
end