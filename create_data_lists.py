from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['/media/ssd/sr data/train2014',
                                     '/media/ssd/sr data/val2014'],
                      test_folders=['/media/ssd/sr data/BSDS100',
                                    '/media/ssd/sr data/Set5',
                                    '/media/ssd/sr data/Set14'],
                      min_size=100,
                      output_folder='./')
