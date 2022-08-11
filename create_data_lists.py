# """"for colab""""
# from utils import create_data_lists

# create_data_lists(voc07_path='/content/learn_computer_vision/pytorch/architecture/SSD/VOC2007',
#                   voc12_path='/content/learn_computer_vision/pytorch/architecture/SSD/VOC2012',
#                   output_folder='/content/learn_computer_vision/pytorch/architecture/SSD')
# print("done create_data_lists.py")


# """"for kaggle""""
# from utils import create_data_lists

# create_data_lists('./AllDatabase',
#                   './')
# print("done create_data_lists.py")

from utils import create_data_lists_split

create_data_lists_split('AllDatabase','/')
print("done create_data_lists.py")
