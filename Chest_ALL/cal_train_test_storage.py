import os


path = './trainAndTest/test/'
count = 0
for dirname in os.listdir(path):
    dir = os.path.join(path, dirname)
    filesize = os.path.getsize(dir)
    count += filesize
print(count/1024.0/1024.0/1024.0)
