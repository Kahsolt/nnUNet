import shutil
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import listdir, join, isdir, maybe_mkdir_p
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def convert_hntsmrg24_task1(hntsmrg_base_dir: str, nnunet_dataset_id: int = 241):
    task_name = "HNTSMRG24_task1"
    folder_name = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, folder_name)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    cnt = 0
    for name in tqdm(listdir(hntsmrg_base_dir)):
        folder_path = join(hntsmrg_base, name)
        if not isdir(folder_path): continue

        shutil.copy(join(folder_path, 'preRT', f'{name}_preRT_T2.nii.gz'),   join(imagestr, f'{name}_0000.nii.gz'))
        shutil.copy(join(folder_path, 'preRT', f'{name}_preRT_mask.nii.gz'), join(labelstr, f'{name}.nii.gz'))
        cnt += 1

    generate_dataset_json(
        out_base, 
        channel_names={
            0: "CT",
        },
        labels={
            'background': 0, 
            'GTVp': 1, 
            'GTVn': 2,
        },
        num_training_cases=cnt, 
        file_ending='.nii.gz',
        dataset_name=task_name, 
        reference='https://hntsmrg24.grand-challenge.org/',
        release='https://zenodo.org/record/11199559',
        overwrite_image_reader_writer='NibabelIOWithReorient',
        description="This is the dataset released in the challenge event. "
                    "It only has the train data for the two tasks.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str,
                        help="The downloaded and extracted HNTSMRG2024 Challenge (https://hntsmrg24.grand-challenge.org/) data. "
                             "Use this link: https://zenodo.org/records/11199559. "
                             "You need to specify the uncompressed root folder of file `HNTSMRG24_train.zip` here!")
    parser.add_argument('-d', required=False, type=int, default=241, help='nnU-Net Dataset ID, default: 241')
    args = parser.parse_args()
    hntsmrg_base = args.input_folder
    convert_hntsmrg24_task1(hntsmrg_base, args.d)
