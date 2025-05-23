import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple, Dict, Callable


class ResNet(nn.Module):
    """Versão modular da ResNet para extração de features.

    Parâmetros
    ----------
    depth : int
        Profundidade da ResNet (18, 34, 50, 101 ou 152).
    pretrained : bool, default=True
        Se True, carrega pesos treinados no ImageNet.
    in_channels : int, default=3
        Número de canais na imagem de entrada (ex.: 3 para RGB, 1 para espectrograma).
    num_classes : Optional[int], default=None
        Se fornecido, adiciona um classifier totalmente conectado.
    global_pool : bool, default=True
        Se True, aplica adaptive average pooling antes da saída.
    return_feats : bool, default=True
        Se True, o forward retorna (features, logits). Se False, retorna apenas logits.
    """

    _constructor_map: Dict[int, Callable] = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    def __init__(
        self,
        depth: int = 50,
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: Optional[int] = None,
        global_pool: bool = True,
        return_feats: bool = True,
    ) -> None:
        super().__init__()
        if depth not in self._constructor_map:
            raise ValueError(f"Profundidade {depth} não suportada. Escolha entre {list(self._constructor_map.keys())}.")

        backbone = self._constructor_map[depth](weights="IMAGENET1K_V1" if pretrained else None)

        # Ajusta a primeira convolução caso o nº de canais seja diferente de 3.
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                if pretrained:
                    if in_channels == 1:
                        # Faz média dos pesos RGB para um único canal
                        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True)) #Copia os pesos da convolução antiga p nova
                    else:
                        repeat = in_channels // 3 #
                        remainder = in_channels % 3
                        new_weight = old_conv.weight.repeat(1, repeat + (1 if remainder else 0), 1, 1)[:, :in_channels, :, :]
                        new_conv.weight.copy_(new_weight / (repeat + (1 if remainder else 0)))
            backbone.conv1 = new_conv

        # Guarda tudo exceto avgpool/fc como extrator de features
        self.feature_extractor = nn.Sequential( #Empilha estrutura padrão da Resnet
            backbone.conv1,  # Convolução inicial 7×7, stride 2 – capta bordas/texturas
            backbone.bn1,    # Normalização por canal 
            backbone.relu,   # Adiciona Não-linearidade com ReLU
            backbone.maxpool,# MaxPool 3×3, stride 2 – reduz resolução (112→56) sem mudar canais
            backbone.layer1, # Stage 2: blocos residuais; mantém 56×56, canais 64→256
            backbone.layer2, # Stage 3: blocos residuais; downsample (stride 2) 56→28, canais 256→512
            backbone.layer3, # Stage 4: blocos residuais; downsample 28→14, canais 512→1024
            backbone.layer4, # Stage 5: blocos residuais; downsample 14→7,  canais 1024→2048
        )
        self.avgpool = backbone.avgpool #Define o avgpool como o padrão do nn.Module (“AdaptiveAvgPool2d(1×1) padrão da ResNet)
        self.global_pool = global_pool #Flag para aplicar pool
        self.return_feats = return_feats #Se é para retornar o número de features
        self.feat_dim = backbone.fc.in_features #Capta a dimensão das features acessando a estrutura do Backbone baseado no nn.Module

        # Classificador opcional
        self.classifier = nn.Linear(self.feat_dim, num_classes) if num_classes else None

    # ---------------------------------------------------------------------
    # Métodos utilitários
    # ---------------------------------------------------------------------
    def freeze_until(self, layer_name: str = "layer2") -> None:
        """Congele gradientes até certo estágio (inclusive)."""
        freeze = True
        for name, param in self.feature_extractor.named_parameters():
            if layer_name in name:
                freeze = False  # libera gradiente a partir daqui
            param.requires_grad = not freeze

    # ---------------------------------------------------------------------
    # Forward (feito "automaticamente" quando o modelo é chamado, 
    # pois __call__ do nn.Module já chama forward() por padrão, estamos
    # apenas sobrescrevendo o método "forward"
    # ---------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, return_feats: Optional[bool] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if return_feats is None:
            return_feats = self.return_feats

        x = self.feature_extractor(x)
        if self.global_pool:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        feats = x
        logits = self.classifier(feats) if self.classifier is not None else None
        return (feats, logits) if return_feats else logits
