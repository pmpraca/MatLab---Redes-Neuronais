function alinea_b()
 img_resolution = [32 32];
 
 imdsT = imageDatastore('train','IncludeSubfolders',1,'LabelSource','foldernames');

 nfich = length(imdsT.Files) % Number of files found
 %imgsRES = zeros(img_resolution(1) * img_resolution(2), nfich);
 
 for i=1:nfich
     
     img = readimage(imdsT,i);
     img = rgb2gray(img);
     % imshow(img)
     img = imresize(img,img_resolution);
     binarizedImg = imbinarize(img);
     
     input(:,i) = reshape(binarizedImg, 1, []);
 end

size(input)

categoricalTargets = imdsT.Labels;
labels = onehotencode(categoricalTargets,2);

target = labels; 
target = target';
fprintf('DISP TARGET')
disp(target)
%size(target)

% Preparar e treinar rede

escolhaRede = 3;
switch escolhaRede
    case 1
        net = feedforwardnet([10]);
        % net = cascadeforwardnet;
        net.trainFcn = 'trainlm'; %trainscg
        net.layers{1}.transferFcn = 'tansig';
        net.layers{2}.transferFcn = 'purelin';
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;
    
    case 2

        net = feedforwardnet([5 5]);

        net.trainFcn = 'trainlm';
        net.layers{1}.transferFcn = 'tansig';
        net.layers{2}.transferFcn = 'tansig';
        net.layers{3}.transferFcn = 'logsig';
        net.divideFcn = 'dividerand';  %divideblock
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;
        
    case 3
        net = feedforwardnet([10]);

        net.trainFcn = 'trainbfg';
        net.layers{1}.transferFcn = 'tansig';
        net.layers{2}.transferFcn = 'purelin';
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        
        net.divideParam.testRatio = 0.15;
        
    case 4
        net = feedforwardnet([10]);

        net.trainFcn = 'trainlm';
        net.layers{1}.transferFcn = 'tansig';
        net.layers{2}.transferFcn = 'purelin';
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;
        
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

plotconfusion(target, out)

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

% SIMULAR A REDE APENAS NO CONJUNTO DE TESTE
  
Tinput = input(:, tr.testInd);
Ttarget = target(:, tr.testInd);

out = sim(net, Tinput);

r = 0;
for i=1:size(out,2)               % Para cada classificacao
    [a b] = max(out(:,i));        % b guarda a linha onde encontrou valor mais alto da saida obtida
    [c d] = max(Ttarget(:,i));     % d guarda a linha onde encontrou valor mais alto da saida desejada
    if b == d                     % se estao na mesma linha, a classificacao foi correta (incrementa 1)
      r = r+1;
    end
end



accuracy = r/size(out,2);
fprintf('Precisao teste %f\n', accuracy)

% Guardar a rede
melhorNet = net;
save(['net1_' num2str(escolhaRede) '.mat'], 'melhorNet');

end