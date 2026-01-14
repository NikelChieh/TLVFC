from torchvision import models as torchmodel
from models import CustomResnet
from toolkit import TLVFC
from toolkit.standardization import FlattenStandardization
from toolkit.matching import IndexMatching
from toolkit.transfer import VarTransfer

# initialize the TLV transfer tool
transfer_tool = TLVFC(
    standardization=FlattenStandardization(),
    matching=IndexMatching(),
    transfer=VarTransfer()
)

# define a pre-trained model and load the checkpoint weight from the torchvision hub
pretrained_model = torchmodel.vgg16(weights = torchmodel.VGG16_Weights.IMAGENET1K_V1)

# define the target model, which is enclosed in a new training mechanism.
target_model = CustomResnet._get_model_custom(model_base='resnet18', num_classes=100)

# start to transfer knowledge
transfer_tool(
    from_module=pretrained_model,
    to_module=target_model
)

# training the target model with knowledge is transferred
train(target_model)