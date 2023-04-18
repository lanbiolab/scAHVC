clear;
 addpath(genpath(pwd));
load('1.mat'); 

cls_num = length(unique(gnd)); %种群数

K = length(X); N = size(X{1},2); %细胞数
for v=1:K
    [X{v}]=NormalizeData(X{v});
end
www=zeros(1,K);
for k=1:K
www(1,k)=1/K;
end
for k=1:K
    Z{k} = zeros(N,N); 
    W{k} = zeros(N,N);
    G{k} = zeros(N,N);
    B{k} = zeros(N,N);
    Q{k} = zeros(N,N);
    L{k} = zeros(N,N);
    E{k} = zeros(size(X{k},1),N);
    Y{k} = zeros(size(X{k},1),N); 
end

dim1 = N;dim2 = N;dim3 = K;
myNorm = 'tSVD_1';
sX = [N, N, K];

Isconverg = 0;epson = 1e-5;
lambda1 =0.01 ; 
lambda2 =0.5; 
yyy=90; 

parOP         =    false;
ABSTOL        =    1e-6;     
RELTOL        =    1e-4;
mu1 = 10e-5; max_mu1 = 10e10; pho_mu1 = 2;
mu2 = 10e-5; max_mu2 = 10e10; pho_mu2 = 2;
rho = 10e-5; max_rho = 10e10; pho_rho = 2;
tic;
start = 1;
for k=1:K
    tmp_inv{k} = inv(2*eye(N,N)+X{k}'*X{k});
end
iter = 0;

while(Isconverg == 0)
    fprintf('----processing iter %d--------\n', iter+1);

    for k=1:K
        if start==1
          Weight{k} = constructW_PKN((abs(Z{k})+abs(Z{k}'))./2, 3);
          Diag_tmp = diag(sum(Weight{k}));
          L{k} = Diag_tmp - Weight{k};
        else

          P =  (abs(Z{k})+abs(Z{k}'))./2;
          param.k = 3;
          HG = gsp_nn_hypergraph(P', param);
          L{k} = HG.L;
        end
        
    end
    start = 0;
    
    for k=1:K
        tmp = (X{k}'*Y{k} + B{k} + mu2*Q{k} - W{k} + rho*G{k})/mu1 + X{k}'*X{k} - X{k}'*E{k};
        Z{k}=tmp_inv{k}*tmp;
    end

    F = [];
    for k=1:K    
        tmp = X{k}-X{k}*Z{k}+Y{k}/mu1;
        F = [F;tmp];
    end
    [Econcat] = solve_l1l2(F,lambda1/mu1);
    start = 1;
    for k=1:K
        E{k} = Econcat(start:start + size(X{k},1) - 1,:);
        start = start + size(X{k},1);
    end
  
    for k=1:K
        Q{k} = (mu2*Z{k} - B{k})*inv(2*www(1,k)*L{k} + mu2*eye(N,N));
    end

    Z_tensor = cat(3, Z{:,:});
    W_tensor = cat(3, W{:,:});
    z = Z_tensor(:);
    w = W_tensor(:); 
    [g, objV] = wshrinkObj(z + 1/rho*w,lambda2/rho,sX,0,3)   ;
    G_tensor = reshape(g, sX);
    
    wwtemp=zeros(1,K);
    wwtemp1=zeros(1,K);
    for k=1:K
        wwtemp(1,k)=trace(Z{k}*L{k}*Z{k}');
    end
%     if iter==0 %计算γ
%         for k=1:K
%             wwtemp1(1,k)=abs(wwtemp(1,k)/log10(www(1,k)));
%         end
% %         yyy=max(wwtemp1);
% yyy=1;
%     end
    for k=1:K
        www(1,k)=exp(-1*wwtemp(1,k)/yyy);
    end   
    totalw=sum(www);
    www=www/totalw;

    w = w + rho*(z - g);
    W_tensor = reshape(w, sX);
    for k=1:K
        Y{k} = Y{k} + mu1*(X{k}-X{k}*Z{k}-E{k});
        B{k} = B{k} + mu2*(Q{k}-Z{k});
    
        G{k} = G_tensor(:,:,k);
        W{k} = W_tensor(:,:,k);
    end
    
    history.objval(iter+1)   =  objV;

    Isconverg = 1;
    for k=1:K
        if (norm(X{k}-X{k}*Z{k}-E{k},inf)>epson)
            history.norm_Z = norm(X{k}-X{k}*Z{k}-E{k},inf);
            fprintf('    norm_Z %7.10f    ', history.norm_Z);
            Isconverg = 0;
        end
        
        if (norm(Z{k}-G{k},inf)>epson)
            history.norm_Z_G = norm(Z{k}-G{k},inf);
            fprintf('norm_Z_G %7.10f    \n', history.norm_Z_G);
            Isconverg = 0;
        end
        error1(k,iter+1) = norm(X{k}-X{k}*Z{k}-E{k},inf);
        error2(k,iter+1) = norm(Z{k}-G{k},inf);

    end
   
    if (iter>200)
        Isconverg  = 1;
    end
    iter = iter + 1;
    mu1 = min(mu1*pho_mu1, max_mu1);
    mu2 = min(mu2*pho_mu2, max_mu2);
    rho = min(rho*pho_rho, max_rho);
end
error1 = mean(error1);
error2 = mean(error2);
S1 = 0;
for k=1:K
    S1 = S1 + abs(Z{k})+abs(Z{k}');
end


C1 = SpectralClustering(S1,cls_num);
imagesc(S1);
[acc,nmi,Fscore,precision,AR,Purity,Recall] = AllMeasure(C1,gnd);
fprintf('ACC %f, NMI %f, Fscore %f,precision %f, AR %f, Purity %f,Recall %f\n', acc,nmi,Fscore,precision,AR,Purity,Recall);
% iters=linspace(1,iter,iter);
% plot(iters,error1,iters,error2);
Y = tsne(S1);
gscatter(Y( : ,1), Y( : ,2),gnd);
toc;

