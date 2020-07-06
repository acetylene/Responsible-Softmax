%% apply the function to a grid on the simplex
%create mesh to graph the simplex. may be joined with code below
X=0:.01:1;
Y=0:.01:1;
Z=zeros(length(X),length(Y));
for i=1:length(X)
for j=1:length(Y)-i
Z(i,j)=abs(1-(X(i)+Y(j)));
end
end

%create the simplex with a mesh
v=zeros(3,5146);
k=1;
for i=0:.001:1
for j=0:.001:1-i
v(:,k)=[i;j;abs(1-i-j)];
k=k+1;
end
end

%set parameters
F=[1,3,3,4;5,1,1,4;3,2,3,3];
G=[F,[1,5;4,2;3,3]];

%look for the fixed points under the iterated map
w=zeros(size(v));
u=zeros(size(v));
for k=1:size(v,2)
w(:,k)=stablepoint(F,v(:,k),9);
u(:,k)=stablepoint(G,v(:,k),9);
end


%% Color each point by it's image
%round to find unique elements of the image
rndu=round(u,4);
rndw=round(w,4);
ImgU=unique(rndu','rows')';%should have 5 columns
ImgW=unique(rndw','rows')';%same as above

Fcolors=zeros(3,size(v,2));
Gcolors=zeros(3,size(v,2));

%identify which point in the image the point in the simplex converges to with G parameters.
for k=1:size(v,2)
diff1=sum(abs(rndu(:,k)-ImgU(:,1)));
diff2=sum(abs(rndu(:,k)-ImgU(:,2)));
diff3=sum(abs(rndu(:,k)-ImgU(:,3)));
diff4=sum(abs(rndu(:,k)-ImgU(:,4)));
diff5=sum(abs(rndu(:,k)-ImgU(:,5)));
if diff2==0
    Gcolors(:,k)=[1,0,0]';
else
    if diff3==0
        Gcolors(:,k)=[0,1,0]';
    else
        if diff4==0
            Gcolors(:,k)=[0,0,1]';
        else 
            if diff5==0
                Gcolors(:,k)=[2,1,0]';
            end
        end
    end
end
end


%same for parameters defined by F
% for k=1:size(v,2)
% diff1=sum(abs(rndw(:,k)-ImgW(:,1)));
% diff2=sum(abs(rndw(:,k)-ImgW(:,2)));
% diff3=sum(abs(rndw(:,k)-ImgW(:,3)));
% diff4=sum(abs(rndw(:,k)-ImgW(:,4)));
% diff5=sum(abs(rndw(:,k)-ImgW(:,5)));
% if diff2==0
%     Fcolors(k)=1;
% else
%     if diff3==0
%         Fcolors(k)=2;
%     else
%         if diff4==0
%             Fcolors(k)=4;
%         else 
%             if diff5==0
%                 Fcolors(k)=8;
%             end
%         end
%     end
% end
% end

%color the mesh defined by X,Y,Z according to the colors found above
Z=squeeze(v(3,:));
Y=squeeze(v(2,:));
X=squeeze(v(1,:));
%scatter3(X,Y,Z,5,Fcolors,'filled')
scatter3(X,Y,Z,5,Gcolors','filled')