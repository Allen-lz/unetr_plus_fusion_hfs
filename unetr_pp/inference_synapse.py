import glob
import os
import SimpleITK as sitk
import numpy as np
import argparse
from medpy import metric
import matplotlib.pyplot as plt

def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())
def hd(pred,gt):
        if pred.sum() > 0 and gt.sum()>0:
            hd95 = metric.binary.hd95(pred, gt)
            return  hd95
        else:
            return 0
            
def process_label(label):
    spleen = label == 1
    right_kidney = label == 2
    left_kidney = label == 3
    gallbladder = label == 4
    liver = label == 6
    stomach = label == 7
    aorta = label == 8
    pancreas = label == 11
   
    return spleen,right_kidney,left_kidney,gallbladder,liver,stomach,aorta,pancreas

def test(fold):
    img_dir = "D:/datasets/UNETR_PP_DATASETS/DATASET/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse/imagesTs"
    label_dir = "D:/datasets/UNETR_PP_DATASETS/DATASET/unetr_pp_raw/unetr_pp_raw_data/Task002_Synapse/labelsTr" # Replace None by full path of "DATASET/unetr_pp_raw/unetr_pp_raw_data/Task002_Synapse/"
    infer_dir = "D:/medical_image/unetr_plus_plus/unetr_pp/evaluation/unetr_pp_synapse_checkpoint/unetr_pp/3d_fullres/Task002_Synapse/unetr_pp_trainer_synapse__unetr_pp_Plansv2.1/fold_0/validation_raw" # Replace None by full path of "output_synapse"

    img_dir = "/home/luzeng/datasets/UNETR_PP_DATASETS/DATASET/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse/imagesTs"
    label_dir = "/home/luzeng/datasets/UNETR_PP_DATASETS/DATASET/unetr_pp_raw/unetr_pp_raw_data/Task002_Synapse/labelsTr"
    infer_dir = "unetr_pp/evaluation/unetr_pp_synapse_checkpoint/unetr_pp/3d_fullres/Task002_Synapse/unetr_pp_trainer_synapse__unetr_pp_Plansv2.1/fold_0/validation_raw"



    # 这里只要先加载infer的list, 之后的label就根据infer的name来取得就行
    infer_list=sorted(glob.glob(os.path.join(infer_dir, '*nii.gz')))
    print("loading success...")

    Dice_spleen=[]
    Dice_right_kidney=[]
    Dice_left_kidney=[]
    Dice_gallbladder=[]
    Dice_liver=[]
    Dice_stomach=[]
    Dice_aorta=[]
    Dice_pancreas=[]
    
    hd_spleen=[]
    hd_right_kidney=[]
    hd_left_kidney=[]
    hd_gallbladder=[]
    hd_liver=[]
    hd_stomach=[]
    hd_aorta=[]
    hd_pancreas=[]
    
    file = infer_dir + 'inferTs/' + fold
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file+'/dice_pre.txt', 'a')
    
    for infer_path in infer_list:
        
        label_path = os.path.join(label_dir, os.path.basename(infer_path).replace("img", "label"))
        img_path = os.path.join(img_dir, os.path.basename(infer_path))

        print(label_path)
        print(img_path)
        print(infer_path)

        # 这里再增加一个原图的预处理
        img, label,infer = read_nii(img_path), read_nii(label_path),read_nii(infer_path)

        # 这里的预处理的结果估计可以用来可视化
        label_spleen,label_right_kidney,label_left_kidney,label_gallbladder,label_liver,label_stomach,label_aorta,label_pancreas=process_label(label)
        infer_spleen,infer_right_kidney,infer_left_kidney,infer_gallbladder,infer_liver,infer_stomach,infer_aorta,infer_pancreas=process_label(infer)

        # for i in range(len(img)):
        #     plt.subplot(131), plt.imshow(np.array(img[i, :, :], dtype=np.uint8))
        #     plt.subplot(132), plt.imshow(np.array(label[i, :, :], dtype=np.uint8))
        #     plt.subplot(133), plt.imshow(np.array(infer[i, :, :], dtype=np.uint8))
        #     plt.show()

        Dice_spleen.append(dice(infer_spleen,label_spleen))
        Dice_right_kidney.append(dice(infer_right_kidney,label_right_kidney))
        Dice_left_kidney.append(dice(infer_left_kidney,label_left_kidney))
        Dice_gallbladder.append(dice(infer_gallbladder,label_gallbladder))
        Dice_liver.append(dice(infer_liver,label_liver))
        Dice_stomach.append(dice(infer_stomach,label_stomach))
        Dice_aorta.append(dice(infer_aorta,label_aorta))
        Dice_pancreas.append(dice(infer_pancreas,label_pancreas))
        
        hd_spleen.append(hd(infer_spleen,label_spleen))
        hd_right_kidney.append(hd(infer_right_kidney,label_right_kidney))
        hd_left_kidney.append(hd(infer_left_kidney,label_left_kidney))
        hd_gallbladder.append(hd(infer_gallbladder,label_gallbladder))
        hd_liver.append(hd(infer_liver,label_liver))
        hd_stomach.append(hd(infer_stomach,label_stomach))
        hd_aorta.append(hd(infer_aorta,label_aorta))
        hd_pancreas.append(hd(infer_pancreas,label_pancreas))
        
        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')
        fw.write('Dice_spleen: {:.4f}\n'.format(Dice_spleen[-1]))
        fw.write('Dice_right_kidney: {:.4f}\n'.format(Dice_right_kidney[-1]))
        fw.write('Dice_left_kidney: {:.4f}\n'.format(Dice_left_kidney[-1]))
        fw.write('Dice_gallbladder: {:.4f}\n'.format(Dice_gallbladder[-1]))
        fw.write('Dice_liver: {:.4f}\n'.format(Dice_liver[-1]))
        fw.write('Dice_stomach: {:.4f}\n'.format(Dice_stomach[-1]))
        fw.write('Dice_aorta: {:.4f}\n'.format(Dice_aorta[-1]))
        fw.write('Dice_pancreas: {:.4f}\n'.format(Dice_pancreas[-1]))
        
        fw.write('hd_spleen: {:.4f}\n'.format(hd_spleen[-1]))
        fw.write('hd_right_kidney: {:.4f}\n'.format(hd_right_kidney[-1]))
        fw.write('hd_left_kidney: {:.4f}\n'.format(hd_left_kidney[-1]))
        fw.write('hd_gallbladder: {:.4f}\n'.format(hd_gallbladder[-1]))
        fw.write('hd_liver: {:.4f}\n'.format(hd_liver[-1]))
        fw.write('hd_stomach: {:.4f}\n'.format(hd_stomach[-1]))
        fw.write('hd_aorta: {:.4f}\n'.format(hd_aorta[-1]))
        fw.write('hd_pancreas: {:.4f}\n'.format(hd_pancreas[-1]))
        
        dsc=[]
        HD=[]
        dsc.append(Dice_spleen[-1])
        dsc.append((Dice_right_kidney[-1]))
        dsc.append(Dice_left_kidney[-1])
        dsc.append(np.mean(Dice_gallbladder[-1]))
        dsc.append(np.mean(Dice_liver[-1]))
        dsc.append(np.mean(Dice_stomach[-1]))
        dsc.append(np.mean(Dice_aorta[-1]))
        dsc.append(np.mean(Dice_pancreas[-1]))
        fw.write('DSC:'+str(np.mean(dsc))+'\n')
        
        HD.append(hd_spleen[-1])
        HD.append(hd_right_kidney[-1])
        HD.append(hd_left_kidney[-1])
        HD.append(hd_gallbladder[-1])
        HD.append(hd_liver[-1])
        HD.append(hd_stomach[-1])
        HD.append(hd_aorta[-1])
        HD.append(hd_pancreas[-1])
        fw.write('hd:'+str(np.mean(HD))+'\n')
        
    
    fw.write('*'*20+'\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_spleen'+str(np.mean(Dice_spleen))+'\n')
    fw.write('Dice_right_kidney'+str(np.mean(Dice_right_kidney))+'\n')
    fw.write('Dice_left_kidney'+str(np.mean(Dice_left_kidney))+'\n')
    fw.write('Dice_gallbladder'+str(np.mean(Dice_gallbladder))+'\n')
    fw.write('Dice_liver'+str(np.mean(Dice_liver))+'\n')
    fw.write('Dice_stomach'+str(np.mean(Dice_stomach))+'\n')
    fw.write('Dice_aorta'+str(np.mean(Dice_aorta))+'\n')
    fw.write('Dice_pancreas'+str(np.mean(Dice_pancreas))+'\n')
    
    fw.write('Mean_hd\n')
    fw.write('hd_spleen'+str(np.mean(hd_spleen))+'\n')
    fw.write('hd_right_kidney'+str(np.mean(hd_right_kidney))+'\n')
    fw.write('hd_left_kidney'+str(np.mean(hd_left_kidney))+'\n')
    fw.write('hd_gallbladder'+str(np.mean(hd_gallbladder))+'\n')
    fw.write('hd_liver'+str(np.mean(hd_liver))+'\n')
    fw.write('hd_stomach'+str(np.mean(hd_stomach))+'\n')
    fw.write('hd_aorta'+str(np.mean(hd_aorta))+'\n')
    fw.write('hd_pancreas'+str(np.mean(hd_pancreas))+'\n')
   
    fw.write('*'*20+'\n')
    
    dsc=[]
    dsc.append(np.mean(Dice_spleen))
    dsc.append(np.mean(Dice_right_kidney))
    dsc.append(np.mean(Dice_left_kidney))
    dsc.append(np.mean(Dice_gallbladder))
    dsc.append(np.mean(Dice_liver))
    dsc.append(np.mean(Dice_stomach))
    dsc.append(np.mean(Dice_aorta))
    dsc.append(np.mean(Dice_pancreas))
    fw.write('dsc:'+str(np.mean(dsc))+'\n')
    
    HD=[]
    HD.append(np.mean(hd_spleen))
    HD.append(np.mean(hd_right_kidney))
    HD.append(np.mean(hd_left_kidney))
    HD.append(np.mean(hd_gallbladder))
    HD.append(np.mean(hd_liver))
    HD.append(np.mean(hd_stomach))
    HD.append(np.mean(hd_aorta))
    HD.append(np.mean(hd_pancreas))
    fw.write('hd:'+str(np.mean(HD))+'\n')
    
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", help="fold name")
    args = parser.parse_args()
    fold=args.fold
    test(fold)
