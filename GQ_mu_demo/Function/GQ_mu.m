function [B,S,time]=GQ_mu(Zm,D,N,par)
fprintf('\n')
fprintf('Start Multispectral Image Unmixing')
fprintf('\n')
t0=tic;
%% Settings
lambda1 = par.lambda1; 
lambda2 = par.lambda2; 
lambda3 = par.lambda3; 
lambda4 = par.lambda4; 
data_name = par.data_name;

%% Initialization
[L1,L2,P]= size(Zm);
Zm       = reshape(Zm,L1*L2,P)';
Zh       = Spectral_Augmentation(Zm,P); % Spectral Augmentation
[A_0,S_0]= HiSun(Zh,N);
[C,W] = simplex_weight(Zh,S_0);

%% Quantum Abundance Tensor (QAT) Extraction
if strcmp(par.QAT, 'Train')

    % ===== Reshape =====
    Zh_tensor = reshape(Zh', L1, L2, []);

    % ===== Path setting =====
    save_pth = 'QNN/QAT/';
    if ~isfolder(save_pth)
        mkdir(save_pth);
    end

    % ===== Save initial data =====
    epoch = 200;
    save(fullfile(save_pth, 'initial.mat'), 'Zh_tensor', 'A_0', 'epoch');

    % ===== Call Python =====
    system('python3 Train_QDIP.py');

    % ===== Load result =====
    load(fullfile(save_pth, 'QNN.mat'));

else

    % ===== Load precomputed result =====
    pth = fullfile('QNN', [data_name, '.mat']);
    load(pth);

end

S_QAT = permute(double(S_QNN),[2,3,1]);
S_QAT = reshape(S_QAT,[],N)';
 %% main altering optimization,
for ii=1:5

    % ===== Update Variable =====
    if ii==1
        A_t=A_0; S_t=S_0;
    else
        A_t=A; S_t=S; lambda4 = lambda4*1.2;
    end
    
    % ===== OA Update =====
    S = Algorithm_2(Zm,Zh,S_QAT,D,A_t,S_t,lambda1,lambda2);
    
    A = updata_A(Zm,Zh,D,S,C,W,A_t,lambda3,lambda4);
end

B=D*A;
time=toc(t0);
return

%% Subprogram 1
function S=Algorithm_2(Zm,Zh,S_Qu,D,A_t,S_t,lambda1,lambda2) % for solving (13)
Y  = S_t;
N  = size(Y,1);
V  = zeros(size(Y));
mu = 1;
cc = lambda1/mu;
R  = A_t'*D'*D*A_t+A_t'*A_t;
P  = A_t'*D'*Zm+A_t'*Zh+lambda2*S_Qu;
I_N = eye(N); 
for i=1:20
    T = Y-V;
    S = inv(R+(lambda2+mu)*I_N)*(P+mu*T); S(S<0)=0;
    Y = shrink(S+V,cc);
    V = V-(Y-S);
end
return

%% Subprogram 2
function A=updata_A(Zm,Zh,D,S,C,W,A_t,lambda3,lambda4)
[M,N]= size(A_t);
I_M  = eye(M);
c_hat= reshape(C,[],1);% C refers to the matrix, c*1_N', in our implementation.
a_t  = reshape(A_t,[],1);
zm   = reshape(Zm,[],1);
zh   = reshape(Zh,[],1);
J4   = kron(W, I_M);
J3   = kron(S, I_M);
J2   = kron(S, D');
J1   = kron(S*S',D'*D)+ kron(S*S',I_M)+ lambda3*J4+ lambda4*eye(M*N);
a_new= inv(J1)*(J2*zm+J3*zh+lambda3*J4*c_hat+lambda4*a_t); a_new(a_new<0)=0;
A    = reshape(a_new,M,N);
return 

%% Shrinkage algroithm
function y=shrink(x,c)
y=zeros(size(x));
y(x>=c)=x(x>=c)-c;
y(abs(x)<=c)=0;
y(x<=-c)=x(x<=-c)+c;
return

%% Obtaining simplex center and weight
function [C,W]=simplex_weight(A,S)
C = mean(A,2)*ones(1,size(S,1));
s = sum(abs(S),2);
s = 1./s;
w = softmax(s,max(s));
W = diag(w);
return
%% Softmax
function y=softmax(x,a)
x = exp(x./a);
s = sum(x,'all');
y = x./s;
return