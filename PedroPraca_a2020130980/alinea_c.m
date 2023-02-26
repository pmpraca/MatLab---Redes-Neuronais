function alinea_c()
%% Ler da pasta test

img_resolution = [32 32];

imdsT = imageDatastore('test','IncludeSubfolders',1,'LabelSource','foldernames');

nfich = length(imdsT.Files); % Number of files found

for i=1:nfich
    
    img = readimage(imdsT,i);
    img = rgb2gray(img);
    %imshow(img)
    img = imresize(img,img_resolution);
    binarizedImg = imbinarize(img);
    
    input_test(:,i) = reshape(binarizedImg, 1, []);
end

categoricalTargets = imdsT.Labels;
labels = onehotencode(categoricalTargets,2);
target_test = labels;
target_test = target_test';


%% Carregar melhor da alinea B

load("netMELHOR.mat");
net = melhorNet;

%% testar test com melhor da alinea b)
out = sim(net, input_test);

plotconfusion(target_test, out);

%Calcula e mostra a percentagem de classificacoes corretas no total dos exemplos
r=0;
for k=1:size(out,2)                 % Para cada classificacao
    [a b] = max(out(:,k));          % b guarda a linha onde encontrou valor mais alto da saida obtida
    [c d] = max(target_test(:,k));  % d guarda a linha onde encontrou valor mais alto da saida desejada
    if b == d                       % se estao na mesma linha, a classificacao foi correta (incrementa 1)
        r = r+1;
    end
end
accuracy = r/size(out,2)*100;
fprintf('Precisao test com melhor B %f\n', accuracy)

%% Testar alinea c) ponto 2 (treinar rede com exemplos pasta 'test')

img_resolution = [32 32];

imdsT = imageDatastore('test','IncludeSubfolders',1,'LabelSource','foldernames');

nfich = length(imdsT.Files); % Number of files found

for i=1:nfich
    
    img = readimage(imdsT,i);
    img = rgb2gray(img);
    %imshow(img)
    img = imresize(img,img_resolution);
    binarizedImg = imbinarize(img);
    
    input_test(:,i) = reshape(binarizedImg, 1, []);
end

categoricalTargets = imdsT.Labels;
labels = onehotencode(categoricalTargets,2);
target_test = labels;
target_test = target_test';

net = feedforwardnet([10]);

net.trainFcn = 'trainlm';
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

[net,tr] = train(net, input_test, target_test);

out = sim(net, input_test);

plotconfusion(target_test, out);
%Calcula e mostra a percentagem de classificacoes corretas no total dos exemplos
r=0;
for k=1:size(out,2)                 % Para cada classificacao
    [a b] = max(out(:,k));          % b guarda a linha onde encontrou valor mais alto da saida obtida
    [c d] = max(target_test(:,k));  % d guarda a linha onde encontrou valor mais alto da saida desejada
    if b == d                       % se estao na mesma linha, a classificacao foi correta (incrementa 1)
        r = r+1;
    end
end
accuracy = r/size(out,2)*100;
fprintf('Precisao test %f\n', accuracy)


%% Save net anterior
save('net_anterior9.mat','net');
load("net_anterior9.mat");
net = melhorNet;
%% Ler da pasta start
imdsT = imageDatastore('start','IncludeSubfolders',1,'LabelSource','foldernames');

nfich = length(imdsT.Files); % Number of files found

for i=1:nfich
    
    
    img = readimage(imdsT,i);
    img = rgb2gray(img);
    %imshow(img)
    img = imresize(img,img_resolution);
    binarizedImg = imbinarize(img);
    
    input_start(:,i) = reshape(binarizedImg, 1, []);
end

categoricalTargets = imdsT.Labels;
labels = onehotencode(categoricalTargets,2);
target_start = labels;
target_start = target_start';

%% testar na start
out = sim(net, input_start);

plotconfusion(target_start, out);
%Calcula e mostra a percentagem de classificacoes corretas no total dos exemplos
r=0;
for k=1:size(out,2)               % Para cada classificacao
    [a b] = max(out(:,k));          %b guarda a linha onde encontrou valor mais alto da saida obtida
    [c d] = max(target_start(:,k));  %d guarda a linha onde encontrou valor mais alto da saida desejada
    if b == d                       % se estao na mesma linha, a classificacao foi correta (incrementa 1)
        r = r+1;
    end
end
accuracy = r/size(out,2)*100;
fprintf('Precisao start %f\n', accuracy)
%% Ler da pasta train

imdsT = imageDatastore('train','IncludeSubfolders',1,'LabelSource','foldernames');

nfich = length(imdsT.Files); % Number of files found

for i=1:nfich
    
    
    img = readimage(imdsT,i);
    img = rgb2gray(img);
    %imshow(img)
    img = imresize(img,img_resolution);
    binarizedImg = imbinarize(img);
    
    input_train(:,i) = reshape(binarizedImg, 1, []);
end

categoricalTargets = imdsT.Labels;
labels = onehotencode(categoricalTargets,2);
target_train = labels;
target_train = target_train';

%% testar na train
out = sim(net, input_train);

plotconfusion(target_train, out);

%Calcula e mostra a percentagem de classificacoes corretas no total dos exemplos
r=0;
for k=1:size(out,2)               % Para cada classificacao
    [a b] = max(out(:,k));          %b guarda a linha onde encontrou valor mais alto da saida obtida
    [c d] = max(target_train(:,k));  %d guarda a linha onde encontrou valor mais alto da saida desejada
    if b == d                       % se estao na mesma linha, a classificacao foi correta (incrementa 1)
        r = r+1;
    end
end
accuracy = r/size(out,2)*100;
fprintf('Precisao train  %f\n', accuracy)

%% Ler da pasta test

imdsT = imageDatastore('test','IncludeSubfolders',1,'LabelSource','foldernames');

nfich = length(imdsT.Files); % Number of files found

for i=1:nfich
    
    
    img = readimage(imdsT,i);
    img = rgb2gray(img);
    %imshow(img)
    img = imresize(img,img_resolution);
    binarizedImg = imbinarize(img);
    
    input_test(:,i) = reshape(binarizedImg, 1, []);
end

categoricalTargets = imdsT.Labels;
labels = onehotencode(categoricalTargets,2);
target_test = labels;
target_test = target_test';

%% testar na test
out = sim(net, input_test);

plotconfusion(target_test, out);
%Calcula e mostra a percentagem de classificacoes corretas no total dos exemplos
r=0;
for k=1:size(out,2)               % Para cada classificacao
    [a b] = max(out(:,k));          %b guarda a linha onde encontrou valor mais alto da saida obtida
    [c d] = max(target_test(:,k));  %d guarda a linha onde encontrou valor mais alto da saida desejada
    if b == d                       % se estao na mesma linha, a classificacao foi correta (incrementa 1)
        r = r+1;
    end
end
accuracy = r/size(out,2)*100;
%fprintf('Precisao test dps de treinar com pasta test %f\n', accuracy)

%%
melhorNet = net;
save('MelhorNET9.mat','melhorNet');

end