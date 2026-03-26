
import os
from PIL import Image



def convert(annot_file, img_width, img_height, new_file):
    #INDEX SPOT IN LINE
    #top left x -> 0
    #top left y -> 1
    #img width -> 2
    #img height -> 3
    #score -> 4
    #category -> 5
    
    with open(annot_file, "r") as f:
        with open(new_file, "w") as f2:

            for l in f:
                line = l.split(",")
                score = int(line[4])
                category = int(line[5])
                if score == 0 or category == 0:
                    continue
                old_width = int(line[2])
                old_height = int(line[3])
                x = (int(line[0]) + old_width / 2)/img_width
                y = (int(line[1]) + old_height / 2)/img_height
                norm_width = old_width/img_width
                norm_height = old_height/img_height

                f2.write(f"{category-1} {x} {y} {norm_width} {norm_height}\n")


def convert_directory(src, path_dest, type="train"):

    
    print(f"Starting processing for img/labels @ {src}")
    #os.path.splitext(filename)[0] stripping extension

    src_images = os.path.join(src, "images")
    src_annots = os.path.join(src, "annotations")

    dest_images = os.path.join(path_dest, "images", type)
    dest_labels = os.path.join(path_dest, "labels", type)

    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(dest_labels, exist_ok=True)

    count = 0
    #extract images 
    for filename in os.listdir(src_images):
        
        #only process .jpg's
        if not filename.endswith(".jpg"): #kip non images
            continue

        
        #open image
        full_img_path = os.path.join(src_images, filename)
        with Image.open(full_img_path) as img:
            #extract image stats
            width, height = img.size
            img_dest = os.path.join(dest_images, filename)

            im_name = os.path.splitext(filename)[0]
        
            match_label_src = os.path.join(src_annots, im_name + ".txt")
            match_label_dest =  os.path.join(dest_labels, im_name + ".txt")

             #look for a matching file in labels. if it exists, create symlink and covert
            if os.path.exists(match_label_src):
                if not os.path.exists(img_dest): #check of dest exists to prevent crashes
                    os.symlink(os.path.abspath(full_img_path), img_dest) #create symlink to dest
                convert(match_label_src, width, height, match_label_dest)
                count += 1

        if count%500 == 0 and count > 1:
            print(f"{count} images processed)")
    
    print(f"final count of images processed: {count}")
    
    
def main():
    #convert images
    train_path = "/Users/sihnat13/dev/projects/droneYOLOv8/datasets/VisDrone2019-DET/VisDrone2019-DET-train"
    val_path = "/Users/sihnat13/dev/projects/droneYOLOv8/datasets/VisDrone2019-DET/VisDrone2019-DET-val"
    dest_path = "datasets/yolo-VisDrone2019-DET"
    convert_directory(train_path, dest_path, type="train")
    convert_directory(val_path, dest_path, type="val")


if __name__ == "__main__":
    main()