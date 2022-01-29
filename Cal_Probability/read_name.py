
# f = open('/mnt/zhengxiaohu/PIRL/outputs/mcs_pre_003/mcs_pre.txt', 'w')
# for a in range(1):
#     for i in range(0, 11000):
#         result = f'mcs_{i}.mat\n'
#         f.write(result)
# f.close()

f = open('/mnt/zhengxiaohu/PIRL/outputs/mcs_pre_003/mcs_pre.txt', 'w')
for a in range(1):
    for i in range(0, 100000):
        result = f'mcs_{i}.mat\n'
        f.write(result)
f.close()
