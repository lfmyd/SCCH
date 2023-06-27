import torch
import torch.nn as nn
from model.SCCH import SCCH


if __name__ == '__main__':
    argparser = SCCH.get_model_specific_argparser()
    hparams = argparser.parse_args()
    torch.cuda.set_device(hparams.device)
    model = SCCH(hparams)

    if hparams.train:
        model.run_training_sessions()
    else:
        model.load()
        print('Loaded model with: %s' % model.flag_hparams())
        model.run_retrieval_case_study()  # retrieval case study
        model.hash_code_visualization()   # hash code visualization