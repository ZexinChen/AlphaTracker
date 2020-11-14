import os

# # data_predictLabel_root =  '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/data_predictLabel04'
# # ori_img_folder_root = '/disk4/zexin/project/mice/datasets/imageFromVideo/'
# # dest_img_root = '/disk4/zexin/datasets/mice/new_labeled_byCompany/04/data_04'

# data_predictLabel_root =  '/disk4/zexin/datasets/mice/new_labeled_byCompany/05/select_merge/'
# ori_img_folder_root = '/disk4/zexin/project/mice/datasets/imageFromVideo/'
# dest_img_root = '/disk4/zexin/datasets/mice/new_labeled_byCompany/05/data_05'

# if not os.path.exists(dest_img_root):
# 	os.mkdir(dest_img_root)

# fns = os.listdir(data_predictLabel_root)
# for fn in fns:
# 	fn_split = fn.split('_')
# 	# print(fn_split)
# 	# print('cp {}/{}/{} {}/'.format(ori_img_folder_root, fn_split[0]+'_'+fn_split[1]+'_'+fn_split[2]+'_'+fn_split[3]+'_'+fn_split[4]+'_' +fn_split[5], fn, dest_img_root))
# 	os.system('cp {}/{}/{} {}/'.format(ori_img_folder_root, fn_split[0]+'_'+fn_split[1]+'_'+fn_split[2]+'_'+fn_split[3]+'_'+fn_split[4]+'_' +fn_split[5], fn, dest_img_root))





ori_img_root = '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/03/color/'
dest_img_root = '/disk4/zexin/datasets/mice/new_labeled_byCompany/0416/03/forDemoVideo/'
if not os.path.exists(dest_img_root):
	os.mkdir(dest_img_root)
for i in range(3000,3600):
	os.system('cp %s/%04d.png %s/'%(ori_img_root,i,dest_img_root))