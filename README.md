# hoop_classifier

get_dataset.py
   
  功能：
  
    得到数据集
  
  var:
  
    hoop_pos：篮筐坐标
    dataset：小图
  
  
  class:
  
    GetHoopPos:得到hoop_pos
    GetDataset：得到dataset
    rough_seperation：粗分
    create_annotation：生成xml
  
  输出：

    1.（后续使用）annotation.xml存放坐标；data.jpg大图
    2. 小图：data.npy存小图数据；label.npy存label、文件夹pos、get存小图



load_dataset.py
  
  功能：
  
    得到data & label，需要HoG处理
    
  输出：
    
    data: list类型，data[0]:阴性样本；data[1]阳性样本
    label: 0阴性, 1阳性  
    
extract_feature.py

   功能：
   
     load_dataset.py 和 HoG.py
     
classifier.py


main.py



    
