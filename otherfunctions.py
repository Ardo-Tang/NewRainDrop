'''
custom functions
'''
def folder_checker(path):
    import os
    import shutil
    if not(os.path.isdir(path)):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

def batch_img_reader(path, datatype="png"):
    from cv2 import cv2
    import os
    def format_checker(name, datatype):
        return (datanamelist[0][-len(datatype)::] == datatype)

    data = []
    datanamelist = os.listdir(path)
    count = 0
    for name in datanamelist:
        count+=1
        print("\r{}/{}{}".format(count,len(datanamelist)," "*10),end="")
        if(format_checker(name, datatype)):
            img = cv2.imread(path+"\\"+name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append(img)
    print("\rRead " + path + " done.")
    return data

def data_shape_normalize(data, shape):
    from cv2 import cv2
    out = data
    for i in range(len(out)):
        out[i] = cv2.resize(data[i], shape, cv2.INTER_LANCZOS4)
    return out

if __name__ == "__main__":
    data = batch_img_reader("D:\\Github\\NewRainDrop\\dataset\\train\\data", "png")
    print(len(data))
    