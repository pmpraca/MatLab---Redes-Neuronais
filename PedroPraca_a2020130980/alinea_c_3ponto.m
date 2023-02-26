function alinea_c_3ponto()
%% treinar com todas as imagens
img_resolution = [32 32];

imdsT = imageDatastore('start_test_train','IncludeSubfolders',1,'LabelSource','foldernames');

nfich = length(imdsT.Files) % Number of files found

for i=1:nfich
    
    img = readimage(imdsT,i);
    img = rgb2gray(img);
    % imshow(img)
    img = imresize(img,img_resolution);
    binarizedImg = imbinarize(img);
    
    input(:,i) = reshape(binarizedImg, 1, []);
end

size(input);

categoricalTargets = imdsT.Labels;
labels = onehotencode(categoricalTargets,2);

target = labels;
target = target';
% disp(target)
size(target);

% Preparar e treinar rede

escolhaRede = 1;
switch escolhaRede
    case 1
        net = feedforwardnet([10]);
        
        net.trainFcn = 'trainlm'; %trainscg
        net.layers{1}.transferFcn = 'purelin';
        net.layers{2}.transferFcn = 'purelin';
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;
        
    case 2
        
        net = feedforwardnet([10 10]);
        
        net.trainFcn = 'trainlm';
        net.layers{1}.transferFcn = 'logsig';
        net.layers{2}.transferFcn = 'tansig';
        net.layers{3}.transferFcn = 'purelin';
        net.divideFcn = 'divideblock';
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;
        
    case 3
        net = feedforwardnet([10]);
        
        net.trainFcn = 'trainlm';
        net.layers{1}.transferFcn = 'purelin';
        net.layers{2}.transferFcn = 'purelin';
        net.divideFcn = 'divideblock';
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;
        
    case 4
        net = feedforwardnet([10]);
        
        net.trainFcn = 'trainlm';
        net.layers{1}.transferFcn = 'purelin';
        net.layers{2}.transferFcn = 'purelin';
        net.divideFcn = 'divideblock';
        net.divideParam.trainRatio = 0.9;
        net.divideParam.valRatio = 0.05;
        net.divideParam.testRatio = 0.05;
        
    case 5
        net = feedforwardnet([20 20 20 20 20 20]);
        
        net.trainFcn = 'trainlm';
        net.layers{1}.transferFcn = 'logsig';
        net.layers{2}.transferFcn = 'purelin';
        net.layers{3}.transferFcn = 'purelin';
        net.layers{4}.transferFcn = 'tansig';
        net.layers{5}.transferFcn = 'logsig';
        net.layers{6}.transferFcn = 'logsig';
        net.layers{7}.transferFcn = 'purelin';
        net.divideFcn = 'divideblock';
        net.divideParam.trainRatio = 0.9;
        net.divideParam.valRatio = 0.05;
        net.divideParam.testRatio = 0.05;
        
    otherwise
        disp('-- Não existe --');
        return;
end
[net,tr] = train(net, input, target);

% Simular e analisar resultados
out = sim(net, input);

plotconfusion(target, out);

disp(tr);

r = 0;
for i=1:size(out,2)               % Para cada classificacao
    [a b] = max(out(:,i));        % b guarda a linha onde encontrou valor mais alto da saida obtida
    [c d] = max(target(:,i));     % d guarda a linha onde encontrou valor mais alto da saida desejada
    if b == d                     % se estao na mesma linha, a classificacao foi correta (incrementa 1)
        r = r+1;
    end
end



accuracy = r/size(out,2);
fprintf('Precisao total de treino %f\n', accuracy)

% Guardar a rede
melhorNet = net;
save(['net_start_train_test_' num2str(escolhaRede) '.mat'], 'melhorNet');

%% testar a rede treinada anterior
load("net_start_train_test_1.mat");
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
fprintf('Precisao test %f\n', accuracy)
melhorNet = net;
save('net_cada_pasta9','melhorNet');
end