
f = open('/mnt/zhengxiaohu_data/dataset_sat_57_center_003_noise/test/test_2.txt', 'w')
# f = open('/mnt/zhengxiaohu_data/datasetD2/test/test.txt', 'w')
# f = open('/mnt/zhengxiaohu_data/dataset_sat_big_57_005_noise/test/test.txt', 'w')
# f = open('/mnt/zhengxiaohu_data/dataset_sat_big_57_005_noise/train/train_val.txt', 'w')
for a in range(1):
    for i in range(5000, 9000):
        result = f'Example{i}.mat\n'
        f.write(result)
f.close()
