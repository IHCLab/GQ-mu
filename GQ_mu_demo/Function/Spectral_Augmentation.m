function  [Zh]= Spectral_Augmentation(Zm,P)
%% BSP function implementation
division=4;
Zh=zeros(P*2,size(Zm,2));
for i=1:P-1
H=(Zm(i+1,:)-Zm(i,:))/division;
Zh(2*i-1,:)=0.5.*(Zm(i,:)-H);
Zh(2*i,:)=0.5.*(Zm(i,:)+H);
end
Zh(2*P-1,:)=0.5.*(Zm(P,:)-H);
Zh(2*P,:)=0.5.*(Zm(P,:)+H);
Zh(Zh<0)=0;
%% PnP denosing
net = denoisingNetwork("DnCNN");
l=sqrt(size(Zh,2));
Zh=Denoise(Zh,l,l,net);
return;

%% PnP for rank
function A=Denoise(x,r,c,net)
A=reshape(x',r,c,[]).*2;
for i=1:size(A,3)
    A(:,:,i) = denoiseImage(A(:,:,i),net);
end
A=reshape(A./2,r*c,[])';
return;