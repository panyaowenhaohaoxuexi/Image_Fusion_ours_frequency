# -*- coding: utf-8 -*-
from net.Network import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction
from net.frequency_fusion import HighLevelGuidedFrequencyFusion
import cv2, numpy as np, torch, warnings, logging, os
import torch.nn as nn
from utils.img_read_save import img_save, image_read_cv2
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path = r'./models/HighLevelGuidedFreqFusion_Clean_latest.pth'
for dataset_name in ["TNO"]:
    print("\n" * 2 + "=" * 80)
    print("The test result of " + dataset_name + ' :')
    path = './test_img/'
    test_folder = os.path.join(path, dataset_name)
    test_out_folder = os.path.join(r'./test_result', dataset_name)
    os.makedirs(test_out_folder, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Restormer_Encoder(inp_channels=1, feature_dim=64)).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder(channels=64, out_channels=1)).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(channels=64)).to(device)
    DetailFuseLayer = nn.DataParallel(HighLevelGuidedFrequencyFusion(in_channels=64, patch_size=4, amp_topk_ratio=0.25, phase_topk_ratio=0.25, token_embed_dim=128, num_heads=4)).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    Encoder.load_state_dict(checkpoint['DIDF_Encoder']); Decoder.load_state_dict(checkpoint['DIDF_Decoder']); BaseFuseLayer.load_state_dict(checkpoint['BaseFuseLayer']); DetailFuseLayer.load_state_dict(checkpoint['DetailFuseLayer'])
    Encoder.eval(); Decoder.eval(); BaseFuseLayer.eval(); DetailFuseLayer.eval()
    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder, 'ir')):
            data_IR = image_read_cv2(os.path.join(test_folder, 'ir', img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
            data_VIS = cv2.split(image_read_cv2(os.path.join(test_folder, 'vi', img_name), mode='YCrCb'))[0][np.newaxis, np.newaxis, ...] / 255.0
            data_VIS_BGR = cv2.imread(os.path.join(test_folder, 'vi', img_name))
            _, data_VIS_Cr, data_VIS_Cb = cv2.split(cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb))
            data_IR = torch.FloatTensor(data_IR).to(device); data_VIS = torch.FloatTensor(data_VIS).to(device)
            feature_V_B, feature_V_F, _ = Encoder(data_VIS)
            feature_I_B, feature_I_F, _ = Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_V_B, feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_F, feature_I_F)
            data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)
            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse) + 1e-8)
            fi = np.squeeze((data_Fuse * 255).cpu().numpy()).astype(np.uint8)
            ycrcb_fi = np.dstack((fi, data_VIS_Cr, data_VIS_Cb))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
            img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)
