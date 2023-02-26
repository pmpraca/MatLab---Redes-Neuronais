function alinea_a()

img_resolution = [32 32];
% nrEpochs = 30;

imdsT = imageDatastore('start','IncludeSubfolders',1,'LabelSource','foldernames'); % acede à pasta especificada incluindo os subfolders
% T = countEachLabel(imdsT)                  % conta todos os labels

nfich = length(imdsT.Files);                 % Number of files found


for i=1:nfich
    
    img = readimage(imdsT,i);                % lê todas as imagens do path indicado por imdsT (image data store)
    img = rgb2gray(img);                     % passar para 625 em vez de 1875 (as imgs agr ficam cinzentas) passa de 3 para 1 camada
    %imshow(img)                             % mostrar as imagens nas pastas (teste para ver se lê tudo)
    img = imresize(img,img_resolution);      % faz o resize de todas as imagens lidas para a resolução [32 32]
    binarizedImg = imbinarize(img);          % cria uma imagem binária e substitui os valores por 1s e os outros valores para 0
    
    input(:,i) = reshape(binarizedImg, 1, []); % faz com que o o array fique numa matriz [1 [] ] ([] -> automaticamente calcula o tamanaho da dimensão
    
    %disp(input)
    
end

size(input)


categoricalTargets = imdsT.Labels;           % Conjunto de labels de dados a codificar, especificado como uma matriz categórica.
                                             % Os elementos dos vetores one-hot encoded correspondem à mesma ordem dada pelas categories(A).

labels = onehotencode(categoricalTargets,2); % Cada elemento de A é substituído por um vector numérico de comprimento igual ao número de classes 
                                             % únicas em A ao longo da dimensão = 2, o vetor contem um 1 na posição de cada classe de categorical Targets
                                             % e 0 nas restantes
                                             
target = labels;                             % zeros(6,30); 
target = target';                            % target = transposta de target 
% fprintfc('DISP TARGET')
% disp(target)
size(target)
%nrEpochs = 30
%for epoch = 1 : nrEpochs
% net = cascadeforwardnet;´
net = feedforwardnet([10]);
net.trainFcn = 'trainscg';
net.layers{1}.transferFcn = 'tansig'; %escondida
net.layers{2}.transferFcn = 'purelin'; %saida
net.divideFcn = '';

[net,tr] = train(net, input, target);

out = sim(net, input);

disp(tr);

r = 0;
for i=1:size(out,2)
    [a b] = max(out(:,i));
    [c d] = max(target(:,i));
    if b == d
        r = r+1;
    end
end

accuracy = r/size(out,2);
fprintf('Precisao total de treino %f\n', accuracy*100)

end
%end
