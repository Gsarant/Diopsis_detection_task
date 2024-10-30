import os
import shutil
import glob
from sklearn.model_selection import KFold
import os, sys
sys.path.append('./misc')
sys.path.append('./')
import tools

def create_k_fold(root_folder,dataset_folder, k_folds):
    list_of_dataset_basename_files=glob.glob(os.path.join(dataset_folder,'annotations','*.json'))
    list_of_dataset_basename_files=[os.path.basename(basenname).split('.')[0] for basenname in list_of_dataset_basename_files]
    kfold = KFold(n_splits=k_folds, shuffle=True)
    for fold, list_kfold_basenames in enumerate(kfold.split(list_of_dataset_basename_files)):
        original_database_images_folder=os.path.join(dataset_folder,'images')
        original_database_annotations_folder=os.path.join(dataset_folder,'annotations')

        kfold_basefolder=os.path.join(root_folder,f"{dataset_folder.split(os.path.sep)[-1]}_{fold}")
        shutil.rmtree(kfold_basefolder, ignore_errors=True)
        tools.create_folders(kfold_basefolder)
        kfold_basefolder_train = os.path.join(kfold_basefolder,'train')
        tools.create_folders(kfold_basefolder_train)
        kfold_basefolder_val = os.path.join(kfold_basefolder,'val')
        tools.create_folders(kfold_basefolder_val)
        
        kfolde_images_folder_train=os.path.join(kfold_basefolder_train,'images')
        kfolde_annotations_folder_train=os.path.join(kfold_basefolder_train,'annotations')
        
        tools.create_folders(kfolde_images_folder_train)
        tools.create_folders(kfolde_annotations_folder_train)
        for index in list_kfold_basenames[0]:
            basename=list_of_dataset_basename_files[index]
            tools.copy_symlink(os.path.abspath(os.path.join(original_database_images_folder,f"{basename}.jpg")),os.path.abspath(os.path.join(kfolde_images_folder_train,f"{basename}.jpg")))
            tools.copy_symlink(os.path.abspath(os.path.join(original_database_annotations_folder,f"{basename}.json")),os.path.abspath(os.path.join(kfolde_annotations_folder_train,f"{basename}.json")))

        kfolde_images_folder_val=os.path.join(kfold_basefolder_val,'images')
        kfolde_annotations_folder_val=os.path.join(kfold_basefolder_val,'annotations')
        tools.create_folders(kfolde_images_folder_val)
        tools.create_folders(kfolde_annotations_folder_val)
        for index in list_kfold_basenames[1]:
            basename=list_of_dataset_basename_files[index]
            tools.copy_symlink(os.path.abspath(os.path.join(original_database_images_folder,f"{basename}.jpg")),os.path.abspath(os.path.join(kfolde_images_folder_val,f"{basename}.jpg")))
            tools.copy_symlink(os.path.abspath(os.path.join(original_database_annotations_folder,f"{basename}.json")),os.path.abspath(os.path.join(kfolde_annotations_folder_val,f"{basename}.json")))
            
        
        

if __name__ == "__main__":
    kfold_dicts=[
        #{'root_folder':"../kfolds/original_diopsis_data_set_train_val", 'dataset_folder':"../original_diopsis_data_set_train_val" , 'k_folds':5},
        #{'root_folder':"../kfolds/original_diopsis_data_set_train_val_synthetic", 'dataset_folder':"../original_diopsis_data_set_train_val_synthetic" , 'k_folds':5}
        {'root_folder':"../kfolds/original_diopsis_data_set_train_val_synthetic_cropted", 'dataset_folder':"../original_diopsis_data_set_train_val_synthetic_cropted" , 'k_folds':5}

    ]
    
    for kfold_dict in kfold_dicts:
        create_k_fold(kfold_dict['root_folder'],kfold_dict['dataset_folder'], kfold_dict['k_folds'])