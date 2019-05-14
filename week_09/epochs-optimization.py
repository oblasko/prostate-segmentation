import os

for epoch in [5,10,15,20,25,30,35,40,45,50]:
    os.system('python epochs-training.py ' + str(epoch))
