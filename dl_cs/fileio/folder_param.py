"""
folder_param

naming convention for training folder
"""


def parameter_to_folder(config):

    num_unrolls = config.MODEL.PARAMETERS.NUM_UNROLLS
    num_resblocks = config.MODEL.PARAMETERS.NUM_RESBLOCKS
    num_emaps = config.MODEL.PARAMETERS.NUM_EMAPS
    num_features = config.MODEL.PARAMETERS.NUM_FEATURES
    weight_loss = config.MODEL.RECON_LOSS.LOSS_WEIGHT

    if weight_loss == True:
        weight_loss = 1
    else:
        weight_loss = 0

    if config.MODEL.MODEL_TYPE == "RES":
        model_type = "resblocks"
    elif config.MODEL.MODEL_TYPE == "SE":
        model_type = "SEblocks"

    return "train-3D_{}steps_{}{}_{}features_{}emaps_{}weight".format(num_unrolls, num_resblocks, model_type, num_features, weight_loss)

def folder_to_parameter(folder_name, write_config, config):

    parts = folder_name.split("_")
    param = {}

    for part in parts:

        #Number of Unrolls
        if part[-5:] == "steps":
            ndig = len(part)-5
            param["num_unrolls"] = int(part[0:ndig])

        #resblocks and number of blocks
        if part[-9:] == "resblocks":
            ndig = len(part)-9
            param["model_type"] = "resblocks"
            param["num_resblocks"] = int(part[0:ndig])

        #resblocks and number of blocks
        if part[-8:] == "SEblocks":
            ndig = len(part)-8
            param["model_type"] = "SEblocks"
            param["num_resblocks"] = int(part[0:ndig])

        #Num Features
        if part[-8:] == "features":
            ndig = len(part)-8
            param["num_features"] = int(part[0:ndig])

        #Weight Loss
        if part[-5:] == "emaps":
            ndig = len(part)-5
            param["num_emaps"] = int(part[0:ndig])

        #Weight Loss
        if part[-6:] == "weight":
            ndig = len(part)-6
            param["loss_weight"] = part[0:ndig] == 1


    if write_config == True:
        config.MODEL.PARAMETERS.NUM_UNROLLS = param["num_unrolls"]
        config.MODEL.PARAMETERS.NUM_RESBLOCKS = param["num_resblocks"]
        config.MODEL.PARAMETERS.NUM_EMAPS = param["num_emaps"]
        config.MODEL.PARAMETERS.NUM_FEATURES = param["num_features"]
        config.MODEL.RECON_LOSS.LOSS_WEIGHT = param["loss_weight"]

    return param
