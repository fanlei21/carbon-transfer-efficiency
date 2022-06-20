% 输入袁文平的litter full数据，使用预测的参数地图，运行凋落物模型，模拟凋落物
%模拟凋落物分解  使用米氏方程的公式模拟
% clc;close all;clear;
format long;  %数据格式15位浮点型
outDir = 'E:\关于凋落物模拟\LIDET\LIRET数据\再处理\outmean\MCMC6_paramMean\机器学习\output全球运行模型\paramBest稳态YeAndGen_litter_MBC_SOCinput_class.csv';
% outDir = 'E:\关于凋落物模拟\LIDET\LIRET数据\再处理\outmean\MCMC6_paramMean\机器学习\output全球运行模型\paramMean稳态YeAndGen_litter_MBC_SOCinput.csv';

%全球参数值
% yefiledir='E:\关于凋落物模拟\LIDET\LIRET数据\再处理\outmean\MCMC6_paramMean\机器学习\output全球\paramMean_grid_预测全球叶参数.csv';
% genfiledir='E:\关于凋落物模拟\LIDET\LIRET数据\再处理\outmean\MCMC6_paramMean\机器学习\output全球\paramMean_grid_预测全球根参数.csv';
% syfiledir='E:\关于凋落物模拟\LIDET\LIRET数据\再处理\outmean\MCMC6_paramMean\机器学习\output全球\paramMean_grid_预测全球有效值索引.csv';

yefiledir='E:\关于凋落物模拟\LIDET\LIRET数据\再处理\outmean\MCMC6_paramMean\机器学习\output全球\paramBest_grid_预测全球叶参数.csv';
genfiledir='E:\关于凋落物模拟\LIDET\LIRET数据\再处理\outmean\MCMC6_paramMean\机器学习\output全球\paramBest_grid_预测全球根参数.csv';
syfiledir='E:\关于凋落物模拟\LIDET\LIRET数据\再处理\outmean\MCMC6_paramMean\机器学习\output全球\paramBest_grid_预测全球有效值索引.csv';
ye_canshu=csvread(yefiledir);
gen_canshu=csvread(genfiledir);
sy_canshu=csvread(syfiledir);

ye_vmax = ye_canshu(:,1);
ye_km = ye_canshu(:,2);
ye_cue = ye_canshu(:,3);
ye_kb = ye_canshu(:,4);
ye_kl = ye_canshu(:,5);

gen_vmax = gen_canshu(:,1);
gen_km = gen_canshu(:,2);
gen_cue = gen_canshu(:,3);
gen_kb = gen_canshu(:,4);
gen_kl = gen_canshu(:,5);

% ye_vmax = ye_canshu(:,1)*(97.9869-1.008)+1.008;
% ye_km = ye_canshu(:,2)*(3980.1036-262.0144)+262.0144;
% ye_cue = ye_canshu(:,3)*(0.29007-0.020024)+0.020024;
% ye_kb = ye_canshu(:,4)*(0.0098515-0.000075771)+0.000075771;
% 
% gen_vmax = gen_canshu(:,1)*(97.9869-1.008)+1.008;
% gen_km = gen_canshu(:,2)*(3980.1036-262.0144)+262.0144;
% gen_cue = gen_canshu(:,3)*(0.29007-0.020024)+0.020024;
% gen_kb = gen_canshu(:,4)*(0.0098515-0.000075771)+0.000075771;

%litter full 凋落物输入
load ('E:\关于凋落物模拟\袁文平Litterfall\cLitter8.mat')

%北美不同的类别
load ('E:\关于凋落物模拟\LIDET\LIRET数据\再处理\outmean\MCMC6_paramMean\机器学习\output全球运行模型\USA\USA_Globcover_CLASS_mask.mat')

% tspan = 0:1:3900;  %变量t的取值范围  步长

% ye_wLM_out=repelem(0,1440*534);
% gen_wLM_out=repelem(0,1440*534);
% ye_wMBC_out=repelem(0,1440*534);
% gen_wMBC_out=repelem(0,1440*534);
% ye_socin_out=repelem(0,1440*534);
% gen_socin_out=repelem(0,1440*534);

hang=size(sy_canshu);
out=repelem(-9999.9,hang(1),9);
for k=1:hang(1)
    sy=sy_canshu(k,1);
    
    class = outdata(sy,1);
    if class==0
        continue;
    end
    
    %农田
    if class==1
        %input 20*20cm   \MCMC6_paramMean的数据已经是m2为单位，不用再/25  下面也不用再乘25
         %假设农田没有枝
        ye_litterfull = cLitter(sy,3)/365.0;
        if ye_litterfull==0
            continue;
        end
        %假设根与地上相等
        gen_litterfull = cLitter(sy,3)/365.0; 
    end
    %阔叶落叶林
    if class==2
        %input 20*20cm   \MCMC6_paramMean的数据已经是m2为单位，不用再/25  下面也不用再乘25
         %假设阔叶落叶林叶茎比为1.98：1
        ye_litterfull = cLitter(sy,3)/(2.98/1.98)/365.0;
        if ye_litterfull==0
            continue;
        end
        %假设根与地上比例为1.34
        gen_litterfull = cLitter(sy,3)*1.34/365.0; 
    end
    %常绿针叶林
    if class==3
        %input 20*20cm   \MCMC6_paramMean的数据已经是m2为单位，不用再/25  下面也不用再乘25
         %假设阔叶落叶林叶茎比为1.94：1
        ye_litterfull = cLitter(sy,3)/(2.94/1.94)/365.0;
        if ye_litterfull==0
            continue;
        end
        %假设根与地上比例为1.26
        gen_litterfull = cLitter(sy,3)*1.26/365.0; 
    end
    %森林灌木草地过度带
    if class==4
        %input 20*20cm   \MCMC6_paramMean的数据已经是m2为单位，不用再/25  下面也不用再乘25
         %假设阔叶落叶林叶茎比为1.89：1
        ye_litterfull = cLitter(sy,3)/(2.89/1.89)/365.0;
        if ye_litterfull==0
            continue;
        end
        %假设根与地上比例为2.08 森林与草原平均值
        gen_litterfull = cLitter(sy,3)*2.08/365.0; 
    end
    %灌木丛
    if class==5
        %input 20*20cm   \MCMC6_paramMean的数据已经是m2为单位，不用再/25  下面也不用再乘25
         %假设阔叶落叶林叶茎比为2.04：1
        ye_litterfull = cLitter(sy,3)/(3.04/2.04)/365.0;
        if ye_litterfull==0
            continue;
        end
        %假设根与地上比例为2.08 森林与草原平均值
        gen_litterfull = cLitter(sy,3)*2.08/365.0; 
    end
    %草原
    if class==6
        %input 20*20cm   \MCMC6_paramMean的数据已经是m2为单位，不用再/25  下面也不用再乘25
         %假设草原没有枝
        ye_litterfull = cLitter(sy,3)/365.0;
        if ye_litterfull==0
            continue;
        end
        %假设根与地上比例为3.65
        gen_litterfull = cLitter(sy,3)*3.65/365.0; 
    end
    
    if ye_kl(k) < 0
        ye_kl(k)=0;
    end
    if gen_kl(k) < 0
        gen_kl(k)=0;
    end
    %稳态值
    ye_wLM = ye_km(k)*ye_kb(k)/(ye_cue(k)*ye_vmax(k)-ye_kb(k));
    gen_wLM = gen_km(k)*gen_kb(k)/(gen_cue(k)*gen_vmax(k)-gen_kb(k));
    %ye_kl = ye_kb(k);
    %ye_wMBC = (litterfull-ye_kl*ye_wLM)*(ye_km(k)+ye_wLM)/(ye_vmax(k)*ye_wLM);
    ye_wMBC = (ye_litterfull-ye_kl(k)*ye_wLM)*(ye_km(k)+ye_wLM)/(ye_vmax(k)*ye_wLM);
    %if ye_wMBC < 0
    %    ye_wMBC = litterfull*(ye_km(k)+ye_wLM)/(ye_vmax(k)*ye_wLM);
    %end
    
    %gen_kl = gen_kb(k);
    %gen_wMBC = (litterfull-gen_kl*gen_wLM)*(gen_km(k)+gen_wLM)/(gen_vmax(k)*gen_wLM);
    gen_wMBC = (gen_litterfull-gen_kl(k)*gen_wLM)*(gen_km(k)+gen_wLM)/(gen_vmax(k)*gen_wLM);
    %if gen_wMBC < 0
    %    gen_wMBC = litterfull*(gen_km(k)+gen_wLM)/(gen_vmax(k)*gen_wLM);
    %end
    
    %SOCinput
    %ye_socin=ye_kl*ye_wLM+ye_kb(k)*ye_wMBC;
    %gen_socin=gen_kl*gen_wLM+gen_kb(k)*gen_wMBC;
    ye_socin=ye_kl(k)*ye_wLM+ye_kb(k)*ye_wMBC;
    gen_socin=gen_kl(k)*gen_wLM+gen_kb(k)*gen_wMBC;
    
    %outdata
    out(k,1)=ye_wLM;
    out(k,2)=gen_wLM;
    out(k,3)=ye_wMBC;
    out(k,4)=gen_wMBC;
    out(k,5)=ye_socin;
    out(k,6)=gen_socin;
    out(k,7)=cLitter(k,1);
    out(k,8)=cLitter(k,2);
    out(k,9)=sy;
    sy
    
end

csvwrite(outDir,out);


% k=1;
% for i=1:1440
%     for j=1:534
%         %input
%         litterfull = cLitter(k,3)/2/365.;
%
%         if litterfull==0
%             k=k+1;
%             continue;
%         end
%
%         if ismember(k,sy_canshu)
%         k
%
%         %稳态值
%         ye_wLM = ye_km(k)*ye_kb(k)/(ye_cue(k)*ye_vmax(k)-ye_kb(k));
%         gen_wLM = gen_km(k)*gen_kb(k)/(gen_cue(k)*gen_vmax(k)-gen_kb(k));
%         ye_kl = ye_kb(k);
%         ye_wMBC = (litterfull-ye_kl*ye_wLM)*(ye_km(k)+ye_wLM)/(ye_vmax(k)*ye_wLM);
%         gen_kl = gen_kb(k);
%         gen_wMBC = (litterfull-gen_kl*gen_wLM)*(gen_km(k)+gen_wLM)/(gen_vmax(k)*gen_wLM);
%
%         %SOCinput
%         ye_socin=ye_kl*ye_wLM+ye_kb(k)*ye_wMBC;
%         gen_socin=gen_kl*gen_wLM+gen_kb(k)*gen_wMBC;
%
%         %outdata
%         out(k,1)=ye_wLM;
%         out(k,2)=gen_wLM;
%         out(k,3)=ye_wMBC;
%         out(k,4)=gen_wMBC;
%         out(k,5)=ye_socin;
%         out(k,6)=gen_socin;
%         out(k,7)=cLitter(k,1);
%         out(k,8)=cLitter(k,2);
%         out(k,9)=k;
%
% %         %初始值？？
% %         x0 = [cLitter(k,3), 0.0023 * 2 * cLitter(k,3)];
% %
% %         ye_k=[ye_vmax(k),ye_km(k),ye_cue(k),ye_kb(k)];
% %         gen_k=[gen_vmax(k),gen_km(k),gen_cue(k),gen_kb(k)];
% %         [~,ye_Xsim]=ode45(@letterfun,tspan,x0,[],ye_k,litterfull);
% %         [~,gen_Xsim]=ode45(@letterfun,tspan,x0,[],gen_k,litterfull);
%
%         end
%         k=k+1;
%     end
% end
%
% csvwrite(outDir,out);
%
%
%
% function y=letterfun(~,x,k,litterfull)
% vmax=k(1);
% km=k(2);
% cue=k(3);
% kb=k(4);
% kl=kb;
%
% y=[litterfull-vmax*x(2)*x(1)/(km+x(1))-kl*x(1),...
%     cue*vmax*x(2)*x(1)/(km+x(1))-kb*x(2)]';
% end