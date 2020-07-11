import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["O_ PE plastic bag", "O_ U-shaped paper clip", "O_ sanitary cup", "O_ Disposable cotton swab", "O_ String of bamboo sticks", "O_ sticky note", "O_ Band aid", "O_ Kitchen gloves", "O_ Mask", "O_ record", "O_ thumbtack", "O_ Prawn head", "O_ Milk teacup", "O_ Dry husk", "O_ desiccant", "O_ Vesicular reticulum", "O_ lighter", "O_ magnifier", "O_ towel", "O_ Altered tape", "O_ Wet wipes", "O_ cigarette end", "O_ toothbrush", "O_ Baijie cloth", "O_ glasses", "O_ bill", "O_ Air conditioning filter element", "O_ Pen and refill", "O_ tissue", "O_ adhesive tap", "O_ Glue waste packaging", "O_ Flyswatter", "O_ Teapot fragments", "O_ Lunch box", "O_ Pregnancy Test Kit", "O_ Feather duster", "K_ mixed congee", "K_ candied gourd on a stick", "K_ Coffee grounds", "K_ Hami melon", "K_ Cherry Tomatoes", "K_ Almonds", "K_ Pistachio", "K_ Common bread", "K_ Chinese chestnut", "K_ jelly", "K_ Walnut", "K_ Pear", "K_ orange", "K_ Leftovers", "K_ hamburger", "K_ pitaya", "K_ Fried chicken", "K_ Roast chicken and duck", "K_ Dried beef", "K_ melon seed", "K_ Sugar cane", "K_ Raw meat", "K_ tomato", "K_ Chinese cabbage", "K_ ternip", "K_ Vermicelli", "K_ Cakes and Pastries", "K_ Red bean", "K_ Sausage (HAM)", "K_ Carrot", "K_ Peanut skin", "K_ Apple", "K_ Tea", "K_ strawberry", "K_ Poached Egg", "K_ pineapple", "K_ Pineapple bag", "K_ jackfruit", "K_ Garlic", "K_ French fries", "K_ Mushroom", "K_ Broad bean", "K_ egg", "K_ Egg Tart", "K_ watermelon rind", "K_ bagel", "K_ Pepper", "K_ dried tangerine peel", "K_ Green vegetables", "K_ Biscuits", "K_ Banana peel", "K_ as closely linked as flesh and blood", "K_ Chicken wings", "R_ table tennis bat", "R_ book", "R_ vacuum cup", "R_ Fresh box", "R_ envelope", "R_ Charging head", "R_ portable battery", "R_ Charging line", "R_ Eight treasure porridge pot", "R_ knife", "R_ Razor blade", "R_ scissors", "R_ Spoon", "R_ Shoulder bag handbag", "R_ card", "R_ Fork", "R_ Deformation toys", "R_ Desk calendar", "R_ Table lamp", "R_ hair drier", "R_ hu la hoop", "R_ globe", "R_ Subway ticket", "R_ Mat", "R_ Plastic bottle", "R_ Plastic basin", "R_ Milk box", "R_ The milk powder pot", "R_ Aluminum cover of milk powder can", "R_ ruler", "R_ Hat", "R_ Abandoned loudspeaker", "R_ handbag", "R_ mobile phone", "R_ Flashlight", "R_ Bracelet", "R_ Inkjet Cartridge", "R_ tyre pump", "R_ Skin care product empty bottle", "R_ newspaper", "R_ slipper", "R_ patch board", "R_ Washboard", "R_ radio", "R_ magnifier", "R_ Cans", "R_ hot pack", "R_ telescope", "R_ Wooden chopping board", "R_ Wooden toys", "R_ Wood comb", "R_ Wooden spatula", "R_ pillow", Recyclable matte"R_ File bag", "R_ Water cup", "R_ Foam box", "R_ Lampshade", "R_ ashtray", "R_ Kettle", "R_ Hot water bottle", "R_ doll", "R_ Glassware", "R_ Glass pot", "R_ Glass ball", "R_ Electric shaver", "R_ Electric curling stick", "R_ Electric toothbrush", "R_ Electric iron", "R_ Remote control", "R_ Circuit board", "R_ boarding pass", "R_ plate", "R_ bowl", "R_ humidifier", "R_ Remote control for air conditioner", "R_ card", "R_ carton", "R_ Cans", "R_ network card", "R_ Earmuff", "R_ headset", "R_ Stud Earring", "R_ Barbie doll", "R_ Tea pot", "R_ Cake box", "R_ bolt drive", "R_ coat hanger", "R_ Socks", "R_ trousers", "R_ Calculator", "R_ stapler", "R_ Microphone", "R_ Shopping paper bag", "R_ Router", "R_ car keys", "R_ Measuring cup", "R_ nail", "R_ clocks and watches", "R_ steel ball", "R_ pot", "R_ lid", "R_ keyboard", "R_ Tweezers", "R_ shoes", "R_ Mat", "R_ mouse", "H_ LED bulb", "H_ Health care bottles", "H_ Oral liquid bottle", "H_ Nail Polish", "H_ Insecticide", "H_ thermometer", "H_ Eyedrop bottle", "H_ Glass lamp", "H_ Battery", "H_ Battery board", "H_ Iodophor empty bottle", "H_ safflower oil", "H_ The button battery", "H_ glue", "H_ Drug packaging", "H_ a pill", "H_ ointment", "H_ Battery", "H_ sphygmomanometer"]

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        convert_annotation(year, image_id)
    list_file.close()

os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")

